# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os, sys
BASEDIR = os.path.dirname(os.path.abspath(__file__))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
from typing import Optional

import carb
import numpy as np
import omni.appwindow  # Contains handle to keyboard
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.robot.policy.examples.robots import H1FlatTerrainPolicy
from isaacsim.storage.native import get_assets_root_path

from isaacsim.core.prims import RigidPrim
from pxr import UsdPhysics, PhysxSchema
import omni.usd

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_srvs.srv import SetBool, Trigger
from nav_msgs.msg import Odometry

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.api.objects import VisualCuboid
import omni.graph.core as og
from isaacsim.core.utils.extensions import enable_extension

# from isaacsim.core.views import ArticulationView
from builtin_interfaces.msg import Time as TimeMsg
from isaacsim.core.api.simulation_context import SimulationContext


num_robots = 11
env_url = "/Isaac/Environments/Grid/default_environment.usd"

enable_extension("isaacsim.ros2.bridge")
# Creating a action graph with ROS component nodes

class SimPublishers(Node):
    def __init__(self):
        super().__init__('sim_publishers')
        self.simtime_publisher_ = self.create_publisher(TimeMsg, '/sim_time', 10)

        self.publisher_odom = []
        for i in range(num_robots):
            self.publisher_odom.append(self.create_publisher(Odometry, f'/G1_{i}/odom', 10))

    def simtime_publish(self, sim_time):
        # advance sim_time
        # split into sec / nanosec
        sec = int(sim_time)
        nanosec = int((sim_time - sec) * 1e9)
        msg = TimeMsg(sec=sec, nanosec=nanosec)
        self.simtime_publisher_.publish(msg)
    
    def pub_odometry(self, pos, quat, lin_ves, ang_vel, robot_id):
        # Create a new Odometry message
        odom_msg = Odometry()
        # Set the header
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = f"G1_{robot_id}/base_link"
        # Set the position
        odom_msg.pose.pose.position.x = float(pos[0])
        odom_msg.pose.pose.position.y = float(pos[1])
        odom_msg.pose.pose.position.z = float(pos[2])
        # Set the orientation
        odom_msg.pose.pose.orientation.w = float(quat[0])
        odom_msg.pose.pose.orientation.x = float(quat[1])
        odom_msg.pose.pose.orientation.y = float(quat[2])
        odom_msg.pose.pose.orientation.z = float(quat[3])
        # Set the linear velocity
        odom_msg.twist.twist.linear.x = float(lin_ves[0])
        odom_msg.twist.twist.linear.y = float(lin_ves[1])
        odom_msg.twist.twist.linear.z = float(lin_ves[2])
        # Set the angular velocity
        odom_msg.twist.twist.angular.x = float(ang_vel[0])
        odom_msg.twist.twist.angular.y = float(ang_vel[1])
        odom_msg.twist.twist.angular.z = float(ang_vel[2])
        # Publish the message
        self.publisher_odom[robot_id].publish(odom_msg)
        

def create_nodes(num_robots):
    nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("PrimService", "isaacsim.ros2.bridge.ROS2ServicePrim"),
    ]

    def robot_nodes(name, type_str):
        return [(f"{name}_Robot_{i}", type_str) for i in range(num_robots)]

    nodes += robot_nodes("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState")
    nodes += robot_nodes("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState")
    nodes += robot_nodes("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController")
    # nodes += robot_nodes("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry")
    nodes += robot_nodes("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry")
    nodes += robot_nodes("PublishRawTransformTree", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree")
    nodes += robot_nodes("PublishRawTransformTree_Odom", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree")
    nodes += robot_nodes("PublishTransformTree", "isaacsim.ros2.bridge.ROS2PublishTransformTree")

    return nodes

def connect_nodes(num_robots):
    def connect_tick(target, no_robot=False):
        if no_robot:
            return [("OnPlaybackTick.outputs:tick", f"{target}.inputs:execIn")]
        return [("OnPlaybackTick.outputs:tick", f"{target}_Robot_{i}.inputs:execIn") for i in range(num_robots)]
    
    def connect_time(target):
        return [("ReadSimTime.outputs:simulationTime", f"{target}_Robot_{i}.inputs:timeStamp") for i in range(num_robots)]

    def connect(source, dest, attr_src, attr_dest):
        return [(f"{source}_Robot_{i}.outputs:{attr_src}", f"{dest}_Robot_{i}.inputs:{attr_dest}") for i in range(num_robots)]

    connections = []
    # Tick connections
    connections += connect_tick("PublishJointState")
    connections += connect_tick("SubscribeJointState")
    connections += connect_tick("ArticulationController")
    # connections += connect_tick("ComputeOdometry")
    connections += connect_tick("PublishRawTransformTree")
    # connections += connect_tick("PublishRawTransformTree_Odom")
    connections += connect_tick("PublishTransformTree")
    connections += connect_tick("PrimService", no_robot=True)

    # Odometry execution
    # connections += [(f"ComputeOdometry_Robot_{i}.outputs:execOut", f"PublishOdometry_Robot_{i}.inputs:execIn") for i in range(num_robots)]

    # Time connections
    connections += connect_time("PublishJointState")
    connections += connect_time("PublishTransformTree")
    connections += connect_time("PublishRawTransformTree")
    connections += connect_time("PublishRawTransformTree_Odom")
    connections += connect_time("PublishOdometry")

    # Controller connections
    connections += connect("SubscribeJointState", "ArticulationController", "jointNames", "jointNames")
    connections += connect("SubscribeJointState", "ArticulationController", "positionCommand", "positionCommand")
    connections += connect("SubscribeJointState", "ArticulationController", "velocityCommand", "velocityCommand")
    connections += connect("SubscribeJointState", "ArticulationController", "effortCommand", "effortCommand")

    # Odometry data connections
    # connections += connect("ComputeOdometry", "PublishOdometry", "angularVelocity", "angularVelocity")
    # connections += connect("ComputeOdometry", "PublishOdometry", "linearVelocity", "linearVelocity")
    # connections += connect("ComputeOdometry", "PublishOdometry", "orientation", "orientation")
    # connections += connect("ComputeOdometry", "PublishRawTransformTree_Odom", "orientation", "rotation")
    # connections += connect("ComputeOdometry", "PublishOdometry", "position", "position")
    # connections += connect("ComputeOdometry", "PublishRawTransformTree_Odom", "position", "translation")
        
    return connections

def set_values(num_robots):
    def set_value(name, attr, value_fn):
        return [(f"{name}_Robot_{i}.inputs:{attr}", value_fn(i)) for i in range(num_robots)]

    setvals = []

    # Controller
    setvals += set_value("ArticulationController", "robotPath", lambda i: f"/World/G1_{i}/pelvis")

    # Joint state
    setvals += set_value("PublishJointState", "targetPrim", lambda i: f"/World/G1_{i}/pelvis")

    # Namespace
    for name in ["PublishJointState", "SubscribeJointState", "PublishOdometry",
                 "PublishRawTransformTree", "PublishRawTransformTree_Odom", "PublishTransformTree"]:
        setvals += set_value(name, "nodeNamespace", lambda i: f"G1_{i}")

    # Odometry
    # setvals += set_value("ComputeOdometry", "chassisPrim", lambda i: f"/World/G1_{i}/pelvis")
    setvals += set_value("PublishRawTransformTree", "parentFrameId", lambda i: "odom")
    setvals += set_value("PublishRawTransformTree", "childFrameId", lambda i: "pelvis")
    # setvals += set_value("PublishOdometry", "chassisFrameId", lambda i: "pelvis")
    # setvals += set_value("PublishOdometry", "odomFrameId", lambda i: "odom")
    # setvals += set_value("PublishRawTransformTree_Odom", "parentFrameId", lambda i: "world")
    # setvals += set_value("PublishRawTransformTree_Odom", "childFrameId", lambda i: "odom")

    # # TF
    # setvals += set_value("PublishTransformTree", "parentPrim", lambda i: f"/World/G1_{i}/pelvis/pelvis")
    # setvals += set_value("PublishTransformTree", "targetPrims", lambda i: f"/World/G1_{i}/pelvis/head_link")
    
    return setvals


# ----------- MAIN -------------
try:
    (graph_handle, _, _, _) = og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: create_nodes(num_robots),
            og.Controller.Keys.SET_VALUES: set_values(num_robots),
            og.Controller.Keys.CONNECT: connect_nodes(num_robots),
        },
    )    
        
except Exception as e:
    print("[Error] " + str(e))
    
  
class G1:
    def __init__(
        self,
        name: str,
        prim_path: str,
        usd_path: str,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ):
        prim = get_prim_at_path(prim_path)
        
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            prim.GetReferences().AddReference(usd_path)

        self.prim = prim
        self.robot = SingleArticulation(prim_path=prim_path,
                                        name=name,
                                        position=position,
                                        orientation=orientation
                                        )

        self.default_position = np.array([0.0, 0.0, 0.0]) if position is None else position
        self.default_orientation = np.array([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation
            
    def initialize(
        self,
        physics_sim_view: omni.physics.tensors.SimulationView = None,
        effort_modes: str = "force",
        control_mode: str = "position",
        set_gains: bool = True,
    ) -> None:
        """
        Initializes the robot and sets up the controller.

        Args:
            physics_sim_view (optional): The physics simulation view.
            effort_modes (str, optional): The effort modes. Defaults to "force".
            control_mode (str, optional): The control mode. Defaults to "position".
            set_gains (bool, optional): Whether to set the joint gains. Defaults to True.
            set_limits (bool, optional): Whether to set the limits. Defaults to True.
            set_articulation_props (bool, optional): Whether to set the articulation properties. Defaults to True.
        """
        self.robot.initialize(physics_sim_view=physics_sim_view)
        self.robot.get_articulation_controller().set_effort_modes(effort_modes)
        self.robot.get_articulation_controller().switch_control_mode(control_mode)

        full_joint_names = self.robot.dof_names
        
        if set_gains:    
            # Define the gain values for each pattern
            joint_patterns = {
                'hip_yaw': (100, 2),   # (kp, kd) for hip_yaw joints
                'hip_roll': (100, 2),  # (kp, kd) for hip_roll joints
                'hip_pitch': (100, 2), # (kp, kd) for hip_pitch joints
                'knee': (150, 4),     # (kp, kd) for knee joints
                'ankle': (40, 2)      # (kp, kd) for ankle joints
            }

            # Initialize lists to store the matching joint names and their respective gains
            matching_joint_names = []
            kps = []
            kds = []
            
            def matching_joint():
                # Loop over the joint patterns to find the matching joints and assign gains
                for pattern, (kp, kd) in joint_patterns.items():
                    # Find joints matching the pattern (e.g., *_hip_yaw_*, *_hip_roll_*, etc.)
                    pattern_matching_joints = [joint for joint in full_joint_names if pattern in joint]
                    
                    # Add the matching joints to the list
                    matching_joint_names.extend(pattern_matching_joints)
                    
                    # Add the corresponding gains to the lists
                    kps.extend([kp] * len(pattern_matching_joints))
                    kds.extend([kd] * len(pattern_matching_joints))
            
            matching_joint()
            self.robot._articulation_view.set_gains(kps=np.array(kps), 
                                                    kds=np.array(kds),
                                                    joint_names=matching_joint_names
                                                    )

        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }

        self.joint_indices = []
        for joint_seq in ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
                'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']:
            assert joint_seq in full_joint_names, f"Joint {joint_seq} not found in robot's joint names"
            self.joint_indices.append(full_joint_names.index(joint_seq))
        self.joint_indices = np.array(self.joint_indices)

        joints_default_position = []
        for joint_name in full_joint_names:
            if joint_name in default_joint_angles.keys():
                joints_default_position.append(default_joint_angles[joint_name])
            else:
                joints_default_position.append(0.0)

        self.robot.set_joints_default_state(np.array(joints_default_position))
        self.robot.set_enabled_self_collisions(True)
        

robots = []
# spawn world
physics_dt = 1 / 400
rendering_dt = 1 / 100
# my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=8 / 200)
my_world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=rendering_dt)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

# spawn warehouse scene
my_world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=10.0,
            dynamic_friction=10.0,
            restitution=0.01,
)

color_names = [
    "Red", "Orange Red", "Orange", "Gold", "Yellow",
    "Chartreuse", "Green", "Turquoise", "Sky Blue", "Blue", "Violet"
]

colors = [
    [1.0, 0.0, 0.0],
    [1.0, 0.27, 0.0],
    [1.0, 0.5, 0.0],
    [1.0, 0.84, 0.0],
    [1.0, 1.0, 0.0],
    [0.5, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [0.0, 0.5, 1.0],
    [0.0, 0.0, 1.0],
    [0.54, 0.17, 0.89]
]

# spawn robot
for i in range(0, num_robots):
    g1 = G1(
        prim_path="/World/G1_" + str(i),
        # prim_path=f"/World/G1_{i}/pelvis",
        name="G1_" + str(i),
        # usd_path='/root/isaac-ros/assets/g1_12dof.usd',
        usd_path=os.path.join(BASEDIR, "g1_12dof/g1_12dof.usd"),
        position=np.array([0, i, 0.8])
    )

    cube = my_world.scene.add(
        VisualCuboid(
            name="visual_cube_" + str(i),
            position=np.array([0, i, 1.5]),
            prim_path="/World/G1_" + str(i) + "/pelvis/Cube",
            size=1.0,
            scale=np.array([0.1, 0.5, 0.1]),
            color=np.array(colors[i]),
        )
    )
    
    robots.append(g1)

my_world.reset()

for robot in robots:
    robot.initialize()


# ---------- ROS2 Node for each robot ----------
class RobotPoseSetterService(Node):
    def __init__(self, robot_id, robot: G1):
        super().__init__(f'robot_pose_setter_{robot_id}')
        self.robot_id = robot_id
        self.robot = robot

        service_name = f'/G1_{robot_id}/set_robot_pose'
        self.srv = self.create_service(Trigger, service_name, self.handle_set_pose)

        self.get_logger().info(f"Service ready on {service_name}")

    def handle_set_pose(self, request, response):
        av = self.robot.robot._articulation_view
        pos = self.robot.default_position[None, :]
        quat = self.robot.default_orientation[None, :]
        
        for _ in range(10):
            # Teleport (local pose can be used if parent does not move)
            av.set_local_poses(translations=pos, orientations=quat)
            av.set_joint_positions(av._default_joints_state.positions)
            av.set_joint_velocities(np.zeros_like(av._default_joints_state.velocities))
            av.set_joint_efforts(np.zeros_like(av._default_joints_state.efforts))
            
            av.set_joint_position_targets(av._default_joints_state.positions)

            my_world.step(render=False)

        self.get_logger().info(f"Teleported robot {self.robot_id} to {pos} | {quat}")

        response.success = True
        response.message = "Pose set successfully"
        return response
        
# Init ROS
rclpy.init()

# # Create nodes per robot
nodes = []
for i in range(num_robots):
    node = RobotPoseSetterService(i, robots[i])
    nodes.append(node)
sim_time_pub = SimPublishers()
nodes.append(sim_time_pub)

# # ---------- Main Simulation Loop ----------

sim_context = SimulationContext()
physics_cnt = 0
while simulation_app.is_running():
    my_world.step(render=True)
    physics_cnt += 1
    # Publish simulation time
    sim_time_pub.simtime_publish(sim_context.current_time)
    # Spin all nodes
    for node in nodes:
        rclpy.spin_once(node, timeout_sec=0.001)
    
    print(f"[API]    sim time = {sim_context.current_time:.3f} s")
    
    # print(sim_time)
    # print(robots[0].robot.dof_names)
    av = robots[0].robot._articulation_view
    # joint_positions = np.array(av.get_joint_positions())[...,robots[0].joint_indices]
    # joint_velocities = np.array(av.get_joint_velocities())[...,robots[0].joint_indices]
    # print("Joint positions: ", joint_positions)
    # print("Joint velocities: ", joint_velocities)

    for i in range(num_robots):
        # Get the linear and angular velocities of the robot
        av = robots[i].robot._articulation_view
        local_pos, local_quat = av.get_local_poses()
        # print("Local pose: ", local_pose)
        linear_velocities = av.get_linear_velocities()  # (, 3)
        angular_velocities = av.get_angular_velocities()  # (, 3)
        # print("Joint positions: ", linear_velocities)
        # print("Joint velocities: ", angular_velocities)
        # Publish the odometry message
        sim_time_pub.pub_odometry(local_pos[0], local_quat[0], np.array(linear_velocities[0]).astype(np.float32), np.array(angular_velocities[0]).astype(np.float32), i)

# # Cleanup
for node in nodes:
    node.destroy_node()

# rclpy.shutdown()
simulation_app.close()
