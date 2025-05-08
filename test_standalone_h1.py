#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) # we can also run as headless.
import argparse

# Isaacsim Packages
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.manipulators.examples.franka import Franka
import omni.graph.core as og
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from h1 import H1FlatTerrainPolicy
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.types import ArticulationAction

# Ros 2 packages
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time as TimeMsg

import numpy as np
import time
parser = argparse.ArgumentParser(description="Define the number of robots.")
parser.add_argument("--num-robots", type=int, default=1, help="Number of robots (default: 1)")
parser.add_argument(
    "--env-url",
    default="/Isaac/Environments/Grid/default_environment.usd",
    required=False,
    help="Path to the environment url",
)
args = parser.parse_args()
print(f"Number of robots: {args.num_robots}")

first_step = True
reset_needed = False
robots = []
enable_extension("isaacsim.ros2.bridge")
from builtin_interfaces.msg import Time

def float_to_ros_time(t: float) -> Time:
    time_msg = Time()
    time_msg.sec = int(t)               # 정수 초
    time_msg.nanosec = int((t % 1) * 1e9)  # 소수점 이하를 나노초로
    return time_msg

def create_nodes(num_robots):
    nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("PrimService", "isaacsim.ros2.bridge.ROS2ServicePrim"),
    ]

    def robot_nodes(name, type_str):
        return [(f"{name}", type_str)]

    nodes += robot_nodes("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState")
    nodes += robot_nodes("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState")
    nodes += robot_nodes("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController")
    nodes += robot_nodes("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry")
    nodes += robot_nodes("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry")
    nodes += robot_nodes("PublishRawTransformTree", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree")
    nodes += robot_nodes("PublishRawTransformTree_Odom", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree")
    nodes += robot_nodes("PublishTransformTree", "isaacsim.ros2.bridge.ROS2PublishTransformTree")

    return nodes

# Creating a action graph with ROS component nodes
try:
    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: # create_nodes(num_robots),
                [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                ("PublishRawTransformTree", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                ("PublishRawTransformTree_Odom", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                ("PublishTransformTree", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ("PrimService", "isaacsim.ros2.bridge.ROS2ServicePrim"),
            ],
            og.Controller.Keys.CONNECT: [
                ('OnPlaybackTick.outputs:tick', 'PublishJointState.inputs:execIn'),
                ('OnPlaybackTick.outputs:tick', 'SubscribeJointState.inputs:execIn'),
                ('OnPlaybackTick.outputs:tick', 'ArticulationController.inputs:execIn'),
                ('OnPlaybackTick.outputs:tick', 'ComputeOdometry.inputs:execIn'),
                ('OnPlaybackTick.outputs:tick', 'PublishRawTransformTree.inputs:execIn'),
                ('OnPlaybackTick.outputs:tick', 'PublishRawTransformTree_Odom.inputs:execIn'),
                ('OnPlaybackTick.outputs:tick', 'PublishTransformTree.inputs:execIn'),
                ('OnPlaybackTick.outputs:tick', 'PrimService.inputs:execIn'),
                ('ComputeOdometry.outputs:execOut', 'PublishOdometry.inputs:execIn'),
                ('ReadSimTime.outputs:simulationTime', 'PublishJointState.inputs:timeStamp'),
                ('ReadSimTime.outputs:simulationTime', 'PublishTransformTree.inputs:timeStamp'),
                ('ReadSimTime.outputs:simulationTime', 'PublishRawTransformTree.inputs:timeStamp'),
                ('ReadSimTime.outputs:simulationTime', 'PublishRawTransformTree_Odom.inputs:timeStamp'),
                ('ReadSimTime.outputs:simulationTime', 'PublishOdometry.inputs:timeStamp'),
                ('SubscribeJointState.outputs:jointNames', 'ArticulationController.inputs:jointNames'),
                ('SubscribeJointState.outputs:positionCommand', 'ArticulationController.inputs:positionCommand'),
                ('SubscribeJointState.outputs:velocityCommand', 'ArticulationController.inputs:velocityCommand'),
                ('SubscribeJointState.outputs:effortCommand', 'ArticulationController.inputs:effortCommand'),
                ('ComputeOdometry.outputs:angularVelocity', 'PublishOdometry.inputs:angularVelocity'),
                ('ComputeOdometry.outputs:linearVelocity', 'PublishOdometry.inputs:linearVelocity'),
                ('ComputeOdometry.outputs:orientation', 'PublishOdometry.inputs:orientation'),
                ('ComputeOdometry.outputs:orientation', 'PublishRawTransformTree_Odom.inputs:rotation'),
                ('ComputeOdometry.outputs:position', 'PublishOdometry.inputs:position'),
                ('ComputeOdometry.outputs:position', 'PublishRawTransformTree_Odom.inputs:translation')],

            og.Controller.Keys.SET_VALUES: [
                # Providing path to /panda robot to Articulation Controller node
                # Providing the robot path is equivalent to setting the targetPrim in Articulation Controller node
                # ("ArticulationController.inputs:usePath", True),      # if you are using an older version of Isaac Sim, you may need to uncomment this line
                ("ArticulationController.inputs:robotPath", "/World/H1_0"),
                ("PublishJointState.inputs:targetPrim", "/World/H1_0"),
                ("PublishJointState.inputs:topicName", "joint_state"),
                # ("Publish")
                
                # new
                # ('PublishJointState.inputs:nodeNamespace', 'H1_0'),
                # ('SubscribeJointState.inputs:nodeNamespace', 'H1_0'),
                # ('PublishOdometry.inputs:nodeNamespace', 'H1_0'),
                # ('PublishRawTransformTree.inputs:nodeNamespace', 'H1_0'),
                # ('PublishRawTransformTree_Odom.inputs:nodeNamespace', 'H1_0'),
                # ('PublishTransformTree.inputs:nodeNamespace', 'H1_0'),
                # ('ComputeOdometry.inputs:chassisPrim', '/World/H1_0'),
                # ('PublishRawTransformTree.inputs:parentFrameId', 'odom'),
                # ('PublishRawTransformTree.inputs:childFrameId', 'pelvis'),
                # ('PublishOdometry.inputs:chassisFrameId', 'pelvis'),
                # ('PublishOdometry.inputs:odomFrameId', 'odom'),
                # ('PublishRawTransformTree_Odom.inputs:parentFrameId', 'world'),
                # ('PublishRawTransformTree_Odom.inputs:childFrameId', 'odom'),
                ('PublishTransformTree.inputs:parentPrim', '/World/H1_0/pelvis'),
                ('PublishTransformTree.inputs:targetPrims', '/World/H1_0/torso_link')
            ],
        },
    )
except Exception as e:
    print(e)
    
class TestROS2Bridge(Node):
    def __init__(self):

        super().__init__("test_ros2bridge")

        # Create the publisher. This publisher will publish a JointState message to the /joint_command topic.
        self.publisher_ = self.create_publisher(JointState, "joint_command", 10)
        self.subscriber_ = self.create_subscription(
            JointState,
            "joint_state",
            self.listener_callback,
            10)
        # self.subscriber = self.create_subscription()

        # Create a JointState message
        self.joint_state = JointState()
        self.joint_state.name = [
            'left_hip_yaw_joint',
            'right_hip_yaw_joint',
            'torso_joint',
            'left_hip_roll_joint',
            'right_hip_roll_joint',
            'left_shoulder_pitch_joint',
            'right_shoulder_pitch_joint',
            'left_hip_pitch_joint',
            'right_hip_pitch_joint',
            'left_shoulder_roll_joint',
            'right_shoulder_roll_joint',
            'left_knee_joint',
            'right_knee_joint',
            'left_shoulder_yaw_joint',
            'right_shoulder_yaw_joint',
            'left_ankle_joint',
            'right_ankle_joint',
            'left_elbow_joint',
            'right_elbow_joint'
        ]

        self.obs = np.zeros(69)
        
        self.latest_msg = None

        num_joints = len(self.joint_state.name)

        # make sure kit's editor is playing for receiving messages
        self.joint_state.position = np.array([0.0] * num_joints, dtype=np.float64).tolist()

        # position control the robot to wiggle around each joint
        self.time_start = time.time()

        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        # self.joint_state.header.stamp = self.get_clock().now().to_msg()
        # self.joint_state.header.stamp = float_to_ros_time(my_world.current_time)

        # breakpoint()
        # action = h1.forward(base_command)
        if h1._policy_counter % h1._decimation == 0:
            if h1._policy_counter > 0:
                self.obs = h1._compute_observation(self.latest_msg)

                self.obs[9:12] = base_command
                self.obs[12:31] = np.array(self.latest_msg.position) - h1.default_pos
                self.obs[31:50] = np.array(self.latest_msg.velocity)
                self.obs[50:] = h1._previous_action
            else:
                self.obs = h1._compute_observation(self.latest_msg)

            # breakpoint()
        
        h1.action = h1._compute_action(self.obs)
        h1._previous_action = h1.action.copy()
        

        action = h1.default_pos + (h1.action * h1._action_scale)
        # self.robot.apply_action(action)
        h1._policy_counter += 1

        joint_position = np.array(action)
        self.joint_state.position = joint_position.tolist()

        # Publish the message to the topic
        self.publisher_.publish(self.joint_state)
        
    def listener_callback(self, msg):
        for i, name in enumerate(msg.name):
            target_position = self.joint_state.position[i]
            measured_position = msg.position[i]
            error = measured_position - target_position

            position = msg.position[i] if i < len(msg.position) else None
            velocity = msg.velocity[i] if i < len(msg.velocity) else None

        # generate the observation from the messages
        self.latest_msg = msg
        # self.obs = h1._compute_observation(base_command)

class SimTimePublisher(Node):
    def __init__(self):
        super().__init__('sim_time_publisher')
        self.publisher_ = self.create_publisher(TimeMsg, '/sim_time', 10)

    def publish(self, sim_time):
        # advance sim_time
        # split into sec / nanosec
        sec = int(sim_time)
        nanosec = int((sim_time - sec) * 1e9)
        msg = TimeMsg(sec=sec, nanosec=nanosec)
        self.publisher_.publish(msg)

my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=1 / 200)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

# spawn warehouse scene
prim = define_prim("/World/Ground", "Xform")
asset_path = assets_root_path + args.env_url
prim.GetReferences().AddReference(asset_path)

# spawn robot
for i in range(0, args.num_robots):
    h1 = H1FlatTerrainPolicy(
        prim_path="/World/H1_" + str(i),
        name="H1_" + str(i),
        usd_path=assets_root_path + "/Isaac/Robots/Unitree/H1/h1.usd",
        position=np.array([0, i, 1.05]),
    )

    robots.append(h1)

my_world.reset()
for robot in robots:
    robot.initialize()

# ros2
rclpy.init(args=None)
ros2_publisher = TestROS2Bridge()
sim_time_pub = SimTimePublisher()
nodes = [ros2_publisher, sim_time_pub]
# robot command
base_command = np.zeros(3)

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    sim_time_pub.publish(my_world.current_time)
    # Spin all nodes
    for node in nodes:
        rclpy.spin_once(node, timeout_sec=0.0001)

    if my_world.is_playing():
        if i >= 0 and i < 80:
            # forward
            base_command = np.array([0.5, 0, 0])
        elif i >= 80 and i < 130:
            # rotate
            base_command = np.array([0.5, 0, 0.5])
        elif i >= 130 and i < 200:
            # side ways
            base_command = np.array([0, 0, 0.5])
        elif i == 200:
            i = 0
        i += 1

simulation_app.close() # close Isaac Sim