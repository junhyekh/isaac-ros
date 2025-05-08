import yaml
import numpy as np
import torch
import time
import copy

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

LEGGED_GYM_ROOT_DIR = 'unitree_rl_gym'
REAL_TIME_FACTOR = 1.0
ROBOT_ID = 0

def get_gravity_orientation(quaternion):
    qx = quaternion[0]
    qy = quaternion[1]
    qz = quaternion[2]
    qw = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


class G1ObservationSubscriber(Node):
    def __init__(self, default_angles, ang_vel_scale, dof_pos_scale, dof_vel_scale, cmd_scale, robot_id=0):
        super().__init__('g1_observation_subscriber')

        self.robot_id = robot_id

        # simulation parameters
        self.period = 0.8           # for phase signal

        self.walking_joint_names = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        ]

        # joint‐related scalars (tweak these!)
        self.default_angles = default_angles
        self.dof_pos_scale = dof_pos_scale
        self.dof_vel_scale = dof_vel_scale
        self.ang_vel_scale = ang_vel_scale
        self.cmd_scale = cmd_scale

        # storage for latest messages
        self.odom_msg = None
        self.joint_msg = None
        self.obs = None
        self.init_joint_msg = None

        # subscriptions
        self.create_subscription(
            Odometry,
            f'/G1_{self.robot_id}/odom',
            self.odom_callback,
            10)
        self.create_subscription(
            JointState,
            f'/G1_{self.robot_id}/joint_states',
            self.joint_callback,
            10)
        
        # publisher
        self.action_pub = self.create_publisher(JointState, f'/G1_{self.robot_id}/joint_command', qos_profile=10)

        # periodic processing

    def odom_callback(self, msg: Odometry):
        self.odom_msg = msg

    def joint_callback(self, msg: JointState):
        if self.init_joint_msg is None:
            self.init_joint_msg = msg
        self.joint_msg = msg

    def compose_observation(self, cmd, prev_action) -> np.ndarray:
        if self.odom_msg is None or self.joint_msg is None:
            return
        # --- 1) angular velocity (body) and scaled ---
        omega = np.array([
            self.odom_msg.twist.twist.angular.x,
            self.odom_msg.twist.twist.angular.y,
            self.odom_msg.twist.twist.angular.z,
        ]) * self.ang_vel_scale

        # --- 2) gravity orientation in body frame ---
        quat = np.array([
            self.odom_msg.pose.pose.orientation.x,
            self.odom_msg.pose.pose.orientation.y,
            self.odom_msg.pose.pose.orientation.z,
            self.odom_msg.pose.pose.orientation.w,
        ])
        gravity_ori = get_gravity_orientation(quat)

        # omega = np.zeros_like(omega)
        # gravity_ori = np.array([0,0,-1]).astype(np.float32)

        # --- 3) joint positions & velocities, scaled & zero‐centered ---
        ros_sec = self.joint_msg.header.stamp.sec
        ros_nanosec = self.joint_msg.header.stamp.nanosec

        joint_names = self.joint_msg.name
        walking_joint_indices = []
        for i, name in enumerate(joint_names):
            if name in self.walking_joint_names:
                walking_joint_indices.append(i)
        walking_joint_indices = np.array(walking_joint_indices)

        qj = copy.deepcopy(np.array(self.joint_msg.position))[walking_joint_indices]
        dqj = copy.deepcopy(np.array(self.joint_msg.velocity))[walking_joint_indices]

        qj = (qj - self.default_angles) * self.dof_pos_scale
        dqj = dqj * self.dof_vel_scale

        # --- 4) phase signal (sin, cos) ---
        t = ros_sec + ros_nanosec * 1e-9
        t = t/REAL_TIME_FACTOR
        phase = (t % self.period) / self.period
        sin_p, cos_p = np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)

        # --- 5) concatenate into one vector ---
        #    [ω(3), gravity_ori(3), qj(36), dqj(36), sin, cos] → length = 80
        obs = np.concatenate([
            omega,
            gravity_ori,
            cmd * self.cmd_scale,
            qj,
            dqj,
            prev_action,
            np.array([sin_p, cos_p])
        ]).astype(np.float32)
        return obs


    def pub_action(self, action: np.ndarray):
        # wait until we have a joint_msg to know the full joint ordering
        if self.joint_msg is None:
            return

        full_names = list(copy.deepcopy(self.joint_msg.name))         # length 37
        full_positions = list(copy.deepcopy(self.init_joint_msg.position)) # current positions
        # full_positions[1] = float(0)
        # full_positions[2] = float(0)

        # # overlay your 12-dim action into the matching slots
        for cmd_idx, joint_name in enumerate(self.walking_joint_names):
            try:
                i = full_names.index(joint_name)
            except ValueError:
                self.get_logger().warn(f"Joint '{joint_name}' not found in full list")
                continue
            full_positions[i] = float(action[cmd_idx])
            # full_positions[i] = float(self.default_angles[cmd_idx])

        # build and publish the message
        msg = JointState()
        # msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.stamp = self.joint_msg.header.stamp
        msg.name = full_names
        msg.position = full_positions
        # optional: zero velocities & efforts, or keep from joint_msg
        msg.velocity = [0.0] * len(full_names)
        msg.effort   = [0.0] * len(full_names)

        self.action_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    with open(f"unitree_rl_gym/deploy/deploy_mujoco/configs/g1.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)
    

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    node = G1ObservationSubscriber(default_angles, ang_vel_scale, dof_pos_scale, dof_vel_scale, cmd_scale, ROBOT_ID)

    # load policy
    policy = torch.jit.load(policy_path)


    try:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        step_start = time.time()

        counter = 0
        while True:
            rclpy.spin_once(node)
            # print(node.obs.shape)
            # print(node.obs)

            counter += 1
            # 50 Hz
            if time.time() - step_start > 0.02*REAL_TIME_FACTOR:
                step_start = time.time()
            # if counter % control_decimation == 0:
                obs = node.compose_observation(cmd, action)
                if obs is None:
                    continue
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)

                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()

                # print(action)

                # node.pub_action(action)

                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles
                node.pub_action(target_dof_pos)


                # print(target_dof_pos)

    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()