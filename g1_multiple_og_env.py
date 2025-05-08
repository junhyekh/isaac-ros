# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

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

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Pose
# from std_srvs.srv import SetBool, Trigger

from isaacsim.core.prims import SingleArticulation
import omni.graph.core as og
from isaacsim.core.utils.extensions import enable_extension

num_robots = 2
env_url = "/Isaac/Environments/Grid/default_environment.usd"

enable_extension("isaacsim.ros2.bridge")
# Creating a action graph with ROS component nodes

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
    nodes += robot_nodes("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry")
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
    connections += connect_tick("ComputeOdometry")
    connections += connect_tick("PublishRawTransformTree")
    connections += connect_tick("PublishRawTransformTree_Odom")
    connections += connect_tick("PublishTransformTree")
    connections += connect_tick("PrimService", no_robot=True)

    # Odometry execution
    connections += [(f"ComputeOdometry_Robot_{i}.outputs:execOut", f"PublishOdometry_Robot_{i}.inputs:execIn") for i in range(num_robots)]

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
    connections += connect("ComputeOdometry", "PublishOdometry", "angularVelocity", "angularVelocity")
    connections += connect("ComputeOdometry", "PublishOdometry", "linearVelocity", "linearVelocity")
    connections += connect("ComputeOdometry", "PublishOdometry", "orientation", "orientation")
    connections += connect("ComputeOdometry", "PublishRawTransformTree_Odom", "orientation", "rotation")
    connections += connect("ComputeOdometry", "PublishOdometry", "position", "position")
    connections += connect("ComputeOdometry", "PublishRawTransformTree_Odom", "position", "translation")
        
    return connections

def set_values(num_robots):
    def set_value(name, attr, value_fn):
        return [(f"{name}_Robot_{i}.inputs:{attr}", value_fn(i)) for i in range(num_robots)]

    setvals = []
    # Controller
    setvals += set_value("ArticulationController", "robotPath", lambda i: f"/World/G1_{i}")

    # Joint state
    setvals += set_value("PublishJointState", "targetPrim", lambda i: f"/World/G1_{i}")

    # Namespace
    for name in ["PublishJointState", "SubscribeJointState", "PublishOdometry",
                 "PublishRawTransformTree", "PublishRawTransformTree_Odom", "PublishTransformTree"]:
        setvals += set_value(name, "nodeNamespace", lambda i: f"G1_{i}")

    # Odometry
    setvals += set_value("ComputeOdometry", "chassisPrim", lambda i: f"/World/G1_{i}")
    setvals += set_value("PublishRawTransformTree", "parentFrameId", lambda i: "odom")
    setvals += set_value("PublishRawTransformTree", "childFrameId", lambda i: "pelvis")
    setvals += set_value("PublishOdometry", "chassisFrameId", lambda i: "pelvis")
    setvals += set_value("PublishOdometry", "odomFrameId", lambda i: "odom")
    setvals += set_value("PublishRawTransformTree_Odom", "parentFrameId", lambda i: "world")
    setvals += set_value("PublishRawTransformTree_Odom", "childFrameId", lambda i: "odom")

    # TF
    setvals += set_value("PublishTransformTree", "parentPrim", lambda i: f"/World/G1_{i}/pelvis")
    setvals += set_value("PublishTransformTree", "targetPrims", lambda i: f"/World/G1_{i}/torso_link")

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
        set_limits: bool = True,
        set_articulation_props: bool = True,
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
        # max_effort, max_vel, stiffness, damping, self.default_pos, self.default_vel = get_robot_joint_properties(
        #     self.policy_env_params, self.robot.dof_names
        # )
        # if set_gains:
        #     self.robot._articulation_view.set_gains(stiffness, damping)
        # if set_limits:
        #     self.robot._articulation_view.set_max_efforts(max_effort)
        #     self.robot._articulation_view.set_max_joint_velocities(max_vel)
        # if set_articulation_props:
        #     self._set_articulation_props()

    # def _set_articulation_props(self) -> None:
    #     """
    #     Sets the articulation root properties from the policy environment parameters.
    #     """
    #     articulation_prop = get_articulation_props(self.policy_env_params)

    #     solver_position_iteration_count = articulation_prop.get("solver_position_iteration_count")
    #     solver_velocity_iteration_count = articulation_prop.get("solver_velocity_iteration_count")
    #     stabilization_threshold = articulation_prop.get("stabilization_threshold")
    #     enabled_self_collisions = articulation_prop.get("enabled_self_collisions")
    #     sleep_threshold = articulation_prop.get("sleep_threshold")

    #     if solver_position_iteration_count not in [None, float("inf")]:
    #         self.robot.set_solver_position_iteration_count(solver_position_iteration_count)
    #     if solver_velocity_iteration_count not in [None, float("inf")]:
    #         self.robot.set_solver_velocity_iteration_count(solver_velocity_iteration_count)
    #     if stabilization_threshold not in [None, float("inf")]:
    #         self.robot.set_stabilization_threshold(stabilization_threshold)
    #     if isinstance(enabled_self_collisions, bool):
    #         self.robot.set_enabled_self_collisions(enabled_self_collisions)
    #     if sleep_threshold not in [None, float("inf")]:
    #         self.robot.set_sleep_threshold(sleep_threshold)
        

robots = []
# spawn world
my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=8 / 200)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

# spawn warehouse scene
prim = define_prim("/World/Ground", "Xform")
asset_path = assets_root_path + env_url
prim.GetReferences().AddReference(asset_path)

# spawn robot
for i in range(0, num_robots):
    g1 = G1(
        prim_path="/World/G1_" + str(i),
        name="G1_" + str(i),
        usd_path=assets_root_path + "/Isaac/Robots/Unitree/G1/g1.usd",
        position=np.array([0, i, 1.05])
    )
    # g1.prim.initialize()
    robots.append(g1)

my_world.reset()

for robot in robots:
    robot.initialize()

# # ---------- ROS2 Node for each robot ----------
# class RobotPoseSetterService(Node):
#     def __init__(self, robot_id, robot: G1):
#         super().__init__(f'robot_pose_setter_{robot_id}')
#         self.robot_id = robot_id
#         self.robot = robot

#         service_name = f'/G1_{robot_id}/set_robot_pose'
#         self.srv = self.create_service(Trigger, service_name, self.handle_set_pose)

#         self.get_logger().info(f"Service ready on {service_name}")

#     def handle_set_pose(self, request, response):
#         av = self.robot.robot._articulation_view
#         pos = self.robot.default_position[None, :]
#         quat = self.robot.default_orientation[None, :]
        
#         # Teleport (local pose can be used if parent does not move)
#         av.set_local_poses(translations=pos, orientations=quat)
#         av.set_joint_positions(av._default_joints_state.positions)
#         av.set_joint_velocities(av._default_joints_state.velocities)
#         av.set_joint_efforts(av._default_joints_state.efforts)

#         self.get_logger().info(f"Teleported robot {self.robot_id} to {pos} | {quat}")

#         response.success = True
#         response.message = "Pose set successfully"
#         return response
        
# # Init ROS
# rclpy.init()

# # Create nodes per robot
# nodes = []
# for i in range(num_robots):
#     node = RobotPoseSetterService(i, robots[i])
#     nodes.append(node)

# # ---------- Main Simulation Loop ----------
while simulation_app.is_running():
    my_world.step(render=True)
#     # Spin all nodes
#     for node in nodes:
#         rclpy.spin_once(node, timeout_sec=0.001)

# # Cleanup
# for node in nodes:
#     node.destroy_node()

# rclpy.shutdown()
simulation_app.close()
