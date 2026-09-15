#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math

# AppLauncher must be initialized before importing Isaac Lab / USD-dependent modules.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Minimal Isaac Lab 3.0 JointWrenchSensor -> ROS 2 demo."
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--physics_dt", type=float, default=1.0 / 120.0)
parser.add_argument("--ros_rate", type=float, default=100.0)
parser.add_argument("--topic", type=str, default="/franka/joint_wrench")
parser.add_argument("--joint_motion_amplitude", type=float, default=0.15,
                    help="Sinusoidal panda_joint7 amplitude [rad].")
parser.add_argument("--joint_motion_frequency", type=float, default=0.25,
                    help="Sinusoidal panda_joint7 frequency [Hz].")
parser.add_argument("--print_period", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after Kit startup.
import torch
import rclpy
from geometry_msgs.msg import WrenchStamped

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


class WrenchPublisher:
    def __init__(self, topic: str):
        rclpy.init(args=None)
        self.node = rclpy.create_node("isaaclab_joint_wrench_publisher")
        self.publisher = self.node.create_publisher(WrenchStamped, topic, 10)

    def publish(self, force_xyz: torch.Tensor, torque_xyz: torch.Tensor, frame_id: str) -> None:
        msg = WrenchStamped()
        # ROS time is used for interoperability with external ROS 2 tools.
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        f = [float(x) for x in force_xyz.detach().cpu()]
        t = [float(x) for x in torque_xyz.detach().cpu()]
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = f
        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = t
        self.publisher.publish(msg)

    def spin_some(self) -> None:
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def shutdown(self) -> None:
        if getattr(self, "node", None) is not None:
            self.node.destroy_node()
            self.node = None
        if rclpy.ok():
            rclpy.shutdown()


@configclass
class WrenchDemoSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
    )

    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(1.6, 1.2, 0.70),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.55)),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.35)),
    # )

    obstacle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TestBlock",
        spawn=sim_utils.CuboidCfg(
            size=(0.20, 0.50, 0.50),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.20, 0.20)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.65, 0.0, 0.60)),
    )

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    joint_wrench = JointWrenchSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        update_period=0.0,
        debug_vis=False,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=2500.0,
            color=(0.8, 0.8, 0.8),
        ),
    )

    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=2500.0,
            color=(0.9, 0.9, 0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, -2.0, 4.0)),
    )


def run_simulation(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    wrench_sensor = scene["joint_wrench"]

    print("\n" + "=" * 80)
    print("[INFO] Franka body names:")
    for i, name in enumerate(robot.body_names):
        print(f"  body[{i:02d}] = {name}")

    print("\n[INFO] Franka joint names:")
    for i, name in enumerate(robot.joint_names):
        print(f"  joint[{i:02d}] = {name}")

    sensor_body_ids, sensor_body_names = wrench_sensor.find_bodies("panda_link7")
    if len(sensor_body_ids) != 1:
        raise RuntimeError(
            f"JointWrenchSensor did not find exactly one panda_link7: "
            f"names={sensor_body_names}, ids={sensor_body_ids}"
        )
    sensor_body_id = int(sensor_body_ids[0])

    joint7_ids, joint7_names = robot.find_joints("panda_joint7")
    if len(joint7_ids) != 1:
        raise RuntimeError(f"Could not find panda_joint7: {joint7_names}")
    joint7_id = int(joint7_ids[0])

    print("\n[INFO] Wrench sensor body:", wrench_sensor.body_names[sensor_body_id])
    print("[INFO] Joint 7:", robot.joint_names[joint7_id])
    print("[INFO] JointWrenchSensor convention:")
    print("  force  : incoming joint reaction force [N]")
    print("  torque : incoming joint reaction torque [N*m]")
    print("  frame  : child-side incoming joint frame")
    print("  point  : child-side joint anchor")
    print("=" * 80 + "\n")

    # Reset robot to the built-in asset's default state.
    default_joint_pos = robot.data.default_joint_pos.torch.clone()
    default_joint_vel = robot.data.default_joint_vel.torch.clone()
    robot.write_joint_position_to_sim_index(position=default_joint_pos, joint_ids=None)
    robot.write_joint_velocity_to_sim_index(velocity=default_joint_vel, joint_ids=None)
    scene.reset()

    # One physics step ensures post-step sensor data exists.
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.cfg.dt)

    ros_pub = WrenchPublisher(args_cli.topic)
    ros_period = 1.0 / max(args_cli.ros_rate, 1e-6)
    next_ros_time = 0.0
    next_print_time = 0.0
    sim_time = 0.0
    env_id = 0  # Publish env 0 only when num_envs > 1.
    frame_id = "panda_link7_joint_incoming"

    print("[INFO] Simulation started.")
    print(f"[INFO] ROS 2 topic: {args_cli.topic}")
    print(f"[INFO] ROS 2 rate : {args_cli.ros_rate:.1f} Hz")
    print("[INFO] A small panda_joint7 sinusoidal target is applied.")
    print("[INFO] Ctrl+C to stop.\n")

    try:
        while simulation_app.is_running():
            # Position target changes smoothly; current joint state is NOT written directly.
            target = default_joint_pos.clone()
            target[:, joint7_id] += (
                args_cli.joint_motion_amplitude
                * math.sin(2.0 * math.pi * args_cli.joint_motion_frequency * sim_time)
            )

            robot.set_joint_position_target_index(target=target, joint_ids=None)
            scene.write_data_to_sim()
            sim.step()
            sim_time += sim.cfg.dt
            scene.update(sim.cfg.dt)

            force = wrench_sensor.data.force.torch[env_id, sensor_body_id]
            torque = wrench_sensor.data.torque.torch[env_id, sensor_body_id]

            if sim_time + 1e-12 >= next_ros_time:
                ros_pub.publish(force, torque, frame_id)
                while next_ros_time <= sim_time:
                    next_ros_time += ros_period

            ros_pub.spin_some()

            if sim_time + 1e-12 >= next_print_time:
                print(
                    f"[Wrench] t={sim_time:8.3f}s | "
                    f"F=[{float(force[0]):9.3f}, {float(force[1]):9.3f}, {float(force[2]):9.3f}] N | "
                    f"T=[{float(torque[0]):9.3f}, {float(torque[1]):9.3f}, {float(torque[2]):9.3f}] N*m"
                )
                next_print_time += max(args_cli.print_period, 1e-3)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received.")
    finally:
        ros_pub.shutdown()


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        dt=args_cli.physics_dt,
        device=args_cli.device,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.2, 2.2, 1.8], target=[0.0, 0.0, 0.6])

    scene_cfg = WrenchDemoSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.5,
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.update(sim.cfg.dt)

    run_simulation(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
