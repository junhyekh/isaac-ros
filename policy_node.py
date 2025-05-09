import numpy as np
import torch
import time
import copy

import sys
import termios
import tty
import select

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Time as TimeMsg

LEGGED_GYM_ROOT_DIR = 'unitree_rl_gym'
ROBOT_ID = 0

def get_key():
    settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())
    select_ready, _, _ = select.select([sys.stdin], [], [], 0.001)
    key = sys.stdin.read(1) if select_ready else None
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

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

class G1ObservationSubscriber(Node):
    def __init__(self, default_angles, robot_id=0):
        super().__init__('g1_observation_subscriber')

        self.robot_id = robot_id

        # simulation parameters
        self.period = 0.8           # for phase signal

        self.walking_joint_names = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        ]

        # joint‐related scalars
        self.ang_vel_scale = 0.25
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.cmd_scale = np.array([2,2,0.25], dtype=np.float32)
        
        self.default_angles = default_angles
        # self.dof_pos_scale = dof_pos_scale
        # self.dof_vel_scale = dof_vel_scale
        # self.ang_vel_scale = ang_vel_scale
        # self.cmd_scale = cmd_scale

        # storage for latest messages
        self.odom_msg = None
        self.joint_msg = None
        self.obs = None
        self.simtime = None
        self.init_simtime = None
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
        self.create_subscription(
            TimeMsg,
            '/sim_time',
            self.simtime_callback,
            10)
        
        # publisher
        self.action_pub = self.create_publisher(JointState, f'/G1_{self.robot_id}/joint_command', 10)

    def odom_callback(self, msg: Odometry):
        self.odom_msg = msg

    def joint_callback(self, msg: JointState):
        if self.init_joint_msg is None:
            self.init_joint_msg = msg
        self.joint_msg = msg
    
    def simtime_callback(self, msg:TimeMsg):
        if self.init_simtime is None:
            self.init_simtime = msg.sec + msg.nanosec * 1e-9
        self.simtime = msg.sec + msg.nanosec * 1e-9 - self.init_simtime

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

        # --- 3) joint positions & velocities, scaled & zero‐centered ---
        joint_names = self.joint_msg.name
        walking_joint_indices = []
        for i, name in enumerate(self.walking_joint_names):
            walking_joint_indices.append(joint_names.index(name))
        walking_joint_indices = np.array(walking_joint_indices)

        qj = np.array(self.joint_msg.position)[walking_joint_indices]
        dqj = np.array(self.joint_msg.velocity)[walking_joint_indices]/180.0*np.pi

        qj = (qj - self.default_angles) * self.dof_pos_scale
        dqj = dqj * self.dof_vel_scale

        # --- 4) phase signal (sin, cos) ---
        t = copy.deepcopy(self.simtime)
        phase = (t % self.period) / self.period
        sin_p, cos_p = np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)

        # --- 5) concatenate into one vector ---
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

        # # overlay your 12-dim action into the matching slots
        for cmd_idx, joint_name in enumerate(self.walking_joint_names):
            try:
                i = full_names.index(joint_name)
            except ValueError:
                self.get_logger().warn(f"Joint '{joint_name}' not found in full list")
                continue
            full_positions[i] = float(action[cmd_idx])

        # build and publish the message
        msg = JointState()
        msg.header.stamp = self.joint_msg.header.stamp
        msg.name = full_names
        msg.position = full_positions
        self.action_pub.publish(msg)

class ResetPoseClient(Node):
    def __init__(self, rid):
        super().__init__('reset_pose_client')
        self.clients_ = {}
        self.futures = {}
        # create a client for each robot id
        srv_name = f'/G1_{rid}/set_robot_pose'
        cli = self.create_client(Trigger, srv_name)
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for service {srv_name}...')
        self.clients_[rid] = cli

    def send_requests(self):
        for rid, cli in self.clients_.items():
            req = Trigger.Request()
            self.futures[rid] = cli.call_async(req)

    def wait_and_report(self):
        # spin until all requests are done
        while rclpy.ok() and not all(f.done() for f in self.futures.values()):
            rclpy.spin_once(self, timeout_sec=0.1)
        # report results
        for rid, future in self.futures.items():
            try:
                res = future.result()
                status = "OK" if res.success else "FAIL"
                self.get_logger().info(f"[Robot {rid}] reset {status}: {res.message}")
            except Exception as e:
                self.get_logger().error(f"[Robot {rid}] service call failed: {e}")

def main(args=None):
    rclpy.init(args=args)

    policy_path = 'assets/weights/pre_train/g1/motion.pt'
    default_angles = np.array([-0.1,  0. ,  0. ,  0.3, -0.2,  0. , -0.1,  0. ,  0. ,  0.3, -0.2, 0. ], dtype=np.float32)
    action_scale = 0.25
    

    # init ROS node
    node = G1ObservationSubscriber(default_angles, ROBOT_ID)
    client = ResetPoseClient(ROBOT_ID)

    # load policy
    policy = torch.jit.load(policy_path)

    try:
        # Close the viewer automatically after simulation_duration wall-seconds.
        keyboard_start = time.time()
        step_start = node.simtime
        client.send_requests()
        client.wait_and_report()
        action = np.zeros(12, dtype=np.float32)
        cmd = np.array([0.5,0,0.0], dtype=np.float32)
        while True:
            if time.time() - keyboard_start > 0.001:
                keyboard_start = time.time()
                key = get_key()
                if key:
                    if key == 'a':
                        cmd = np.array([0, 0.5, 0], dtype=np.float32)
                    elif key == 'w':
                        cmd = np.array([0.5, 0, 0], dtype=np.float32)
                    elif key == 's':
                        cmd = np.array([-0.5, 0, 0], dtype=np.float32)
                    elif key == 'd':
                        cmd = np.array([0, -0.5, 0], dtype=np.float32)
                    elif key == 'q':
                        cmd = np.array([0, 0, 0.5], dtype=np.float32)
                    elif key == 'e':
                        cmd = np.array([0, 0, -0.5], dtype=np.float32)
                    elif key == 'x':
                        cmd = np.array([0, 0, 0], dtype=np.float32)
                    elif key == '\x03':
                        break  # Ctrl-C to exit loop
                    print(f'Key pressed: {key} -> cmd: {cmd}')
            
            rclpy.spin_once(node)
            if node.simtime is None:
                continue
            if step_start is None:
                step_start = node.simtime
            
            # 50 Hz
            if node.simtime - step_start > 0.02:
                step_start = node.simtime
                obs = node.compose_observation(cmd, action)
                if obs is None:
                    continue
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)

                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()

                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles
                node.pub_action(target_dof_pos)

    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()