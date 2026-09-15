#!/usr/bin/env python3
"""
Isaac Lab 3.0 + ROS 2 + JointWrenchSensor minimal verification.

Scene generated entirely in the script:
  - ground plane only (no table)
  - built-in Isaac Lab Franka Panda
  - one dynamic cube (gravity enabled, non-kinematic)

Two wrench interfaces are published:
  1) geometry_msgs/msg/WrenchStamped
     /franka/joint_wrench
  2) six std_msgs/msg/Float64 scalar topics for rqt_plot
     /franka/joint_wrench/fx
     /franka/joint_wrench/fy
     /franka/joint_wrench/fz
     /franka/joint_wrench/tx
     /franka/joint_wrench/ty
     /franka/joint_wrench/tz

The JointWrenchSensor reports incoming joint reaction wrench in the
INCOMING_JOINT_FRAME convention: child-side joint frame and child-side
joint anchor as torque reference point.
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab JointWrenchSensor ROS2 demo")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--physics_dt", type=float, default=1.0 / 120.0)
parser.add_argument("--ros_rate", type=float, default=100.0)
parser.add_argument("--topic", type=str, default="/franka/joint_wrench")
parser.add_argument("--body_name", type=str, default="panda_link7")
parser.add_argument("--motion_joint", type=str, default="panda_joint4")
parser.add_argument("--motion_amplitude", type=float, default=0.65, help="rad")
parser.add_argument("--motion_frequency", type=float, default=0.20, help="Hz")
# parser.add_argument("--drop_height", type=float, default=0, help="m")
parser.add_argument("--collision_offset_x", type=float, default=0.18, help="m")
parser.add_argument("--collision_offset_z", type=float, default=-0.12, help="m")
parser.add_argument("--collision_size", type=float, default=0.30, help="m")
parser.add_argument("--collision_mass", type=float, default=5.0, help="kg")
# parser.add_argument("--drop_duration", type=float, default=1.5, help="s")
parser.add_argument("--episode_steps", type=int, default=900, help="physics steps per episode; scene resets after this many steps")
parser.add_argument("--reset_pause_steps", type=int, default=5, help="physics steps held after reset before active episode")
parser.add_argument("--print_period", type=float, default=0.25, help="s")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports that depend on Kit/Isaac Sim.
import rclpy
import torch
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


class WrenchRosPublisher:
    """Publish semantic WrenchStamped plus scalar topics for rqt_plot."""

    def __init__(self, base_topic: str):
        rclpy.init(args=None)
        self.node = rclpy.create_node("isaaclab_joint_wrench_publisher")
        self.wrench_pub = self.node.create_publisher(WrenchStamped, base_topic, 10)
        self.scalar_pubs = {
            name: self.node.create_publisher(Float64, f"{base_topic}/{name}", 10)
            for name in ("fx", "fy", "fz", "tx", "ty", "tz")
        }

    def publish(self, force: torch.Tensor, torque: torch.Tensor, frame_id: str) -> None:
        vals = [
            float(force[0]), float(force[1]), float(force[2]),
            float(torque[0]), float(torque[1]), float(torque[2]),
        ]

        msg = WrenchStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.wrench.force.x = vals[0]
        msg.wrench.force.y = vals[1]
        msg.wrench.force.z = vals[2]
        msg.wrench.torque.x = vals[3]
        msg.wrench.torque.y = vals[4]
        msg.wrench.torque.z = vals[5]
        self.wrench_pub.publish(msg)

        for name, value in zip(("fx", "fy", "fz", "tx", "ty", "tz"), vals, strict=True):
            scalar = Float64()
            scalar.data = value
            self.scalar_pubs[name].publish(scalar)

    def spin_some(self) -> None:
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def shutdown(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        if rclpy.ok():
            rclpy.shutdown()


@configclass
class WrenchDemoSceneCfg(InteractiveSceneCfg):
    """Ground + Franka + one genuinely dynamic test object."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
    )

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    collision_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CollisionObject",
        spawn=sim_utils.CuboidCfg(
            size=(args_cli.collision_size,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.collision_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,
                rest_offset=0.005,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.15, 0.15),
                roughness=0.6,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.collision_size / 2.0 + 0.01)),
    )

    joint_wrench = JointWrenchSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        update_period=0.0,
        # This is the only convention currently exposed by JointWrenchSensor.
        convention="incoming_joint_frame",
        debug_vis=False,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=2500.0,
            color=(0.8, 0.8, 0.8),
        ),
    )


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        dt=args_cli.physics_dt,
        device=args_cli.device,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.0, 2.0, 1.6), target=(0.2, 0.0, 0.5))

    scene_cfg = WrenchDemoSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.5,
        replicate_physics=True,
    )
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.update(sim.cfg.dt)

    robot = scene["robot"]
    collision_object = scene["collision_object"]
    wrench_sensor = scene["joint_wrench"]

    print("\n" + "=" * 80)
    print("[INFO] Franka bodies:")
    for i, name in enumerate(robot.body_names):
        print(f"  body[{i:02d}] = {name}")
    print("\n[INFO] Franka joints:")
    for i, name in enumerate(robot.joint_names):
        print(f"  joint[{i:02d}] = {name}")

    body_ids, body_names = wrench_sensor.find_bodies(args_cli.body_name)
    if len(body_ids) != 1:
        raise RuntimeError(
            f"JointWrenchSensor body lookup failed for '{args_cli.body_name}': "
            f"names={body_names}, ids={body_ids}"
        )
    sensor_body_id = int(body_ids[0])

    motion_ids, motion_names = robot.find_joints(args_cli.motion_joint)
    if len(motion_ids) != 1:
        raise RuntimeError(
            f"Motion joint lookup failed for '{args_cli.motion_joint}': "
            f"names={motion_names}, ids={motion_ids}"
        )
    motion_joint_id = int(motion_ids[0])

    print("\n[INFO] Selected wrench body:", args_cli.body_name, sensor_body_id)
    print("[INFO] Selected motion joint:", args_cli.motion_joint, motion_joint_id)
    print("[INFO] Wrench convention: incoming_joint_frame")
    print(f"[INFO] Episode reset period: {args_cli.episode_steps} physics steps")
    print(f"[INFO] Reset pause: {args_cli.reset_pause_steps} physics steps")
    print("[INFO] No table is created.")
    print("[INFO] CollisionObject: kinematic=False, gravity enabled.")
    print("=" * 80 + "\n")

    default_q = robot.data.default_joint_pos.torch.clone()
    default_qd = robot.data.default_joint_vel.torch.clone()

    ros = WrenchRosPublisher(args_cli.topic)
    frame_id = f"{args_cli.body_name}_incoming_joint"

    ros_period = 1.0 / max(args_cli.ros_rate, 1e-6)
    next_ros = 0.0
    next_print = 0.0
    sim_time = 0.0
    episode_index = 0

    # Cache the robot's nominal hand position after a clean reset. This is used
    # to place the collision object reproducibly for every episode.
    def reset_episode() -> None:
        nonlocal next_print

        # InteractiveScene.reset() resets all registered assets and sensors.
        # The official scene implementation calls reset() on articulations,
        # rigid objects and sensors. We then explicitly write the robot state
        # and the test-object pose to the simulator for deterministic restart.
        scene.reset()

        robot.write_joint_state_to_sim(default_q, default_qd)
        robot.set_joint_position_target(default_q)

        # Update once so the FK/body pose corresponds to the reset robot.
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.cfg.dt)

        hand_pos = robot.data.body_pos_w[0, sensor_body_id].detach().clone()

        # Dynamic object starts above the hand; it will fall under gravity.
        # collision_pos = torch.tensor([
        #                             0.65,
        #                             0.0,
        #                             args_cli.collision_size / 2.0 + 0.01,
        #                         ])
        # drop_pos = hand_pos.clone()

        # drop_pos[2] += args_cli.drop_height

        # drop_pose = torch.zeros(
        #     (args_cli.num_envs, 7), device=collision_object.device
        # )
        # drop_pose[:, 3] = args_cli.collision_size / 2.0 + 0.01
        # # All environments use their respective environment-relative hand
        # # position. With num_envs=1 this is simply the world pose above.
        # drop_pose[0, :3] = drop_pos
        # if args_cli.num_envs > 1:
        #     # For additional envs, use each env's hand position.
        #     all_hand_pos = robot.data.body_pos_w[:, sensor_body_id].detach().clone()
        #     drop_pose[:, :3] = all_hand_pos
        #     drop_pose[:, 2] += args_cli.drop_height

        # collision_object.write_root_pose_to_sim(drop_pose)
        # collision_object.write_root_velocity_to_sim(
        #     torch.zeros((args_cli.num_envs, 6), device=collision_object.device)
        # )
        # collision_object.write_data_to_sim()

        # sim.step()
        # scene.update(sim.cfg.dt)
        # -------------------------------------------------------------
        # Reset object directly on the ground.
        # No dropping / throwing motion.
        # -------------------------------------------------------------
        collision_pos = torch.tensor(
            [
                0.65,
                0.0,
                args_cli.collision_size / 2.0 + 0.01,
            ],
            device=collision_object.device,
            dtype=torch.float32,
        )

        collision_pose = torch.zeros(
            (args_cli.num_envs, 7),
            device=collision_object.device,
        )

        collision_pose[:, 3] = 1.0  # identity quaternion
        collision_pose[:, :3] = collision_pos

        collision_object.write_root_pose_to_sim(collision_pose)

        collision_object.write_root_velocity_to_sim(
            torch.zeros(
                (args_cli.num_envs, 6),
                device=collision_object.device,
            )
        )

        collision_object.write_data_to_sim()

        sim.step()
        scene.update(sim_utils.sim.cfg.dt)

        print(
            f"\n[RESET] episode={episode_index} | "
            f"object_drop_z={float(drop_pose[0, 2]):.3f} m"
        )
        next_print = sim_time

    try:
        reset_episode()

        while simulation_app.is_running():
            # -------------------------------------------------------------
            # Phase 1: object drops for a fixed number of physics steps.
            # -------------------------------------------------------------
            # drop_steps = min(
            #     int(round(args_cli.drop_duration / sim.cfg.dt)),
            #     args_cli.episode_steps,
            # )

            # while simulation_app.is_running() and drop_steps > 0:
            #     robot.set_joint_position_target(default_q)
            #     scene.write_data_to_sim()
            #     sim.step()
            #     sim_time += sim.cfg.dt
            #     episode_steps_done = 1
            #     drop_steps -= 1
            #     scene.update(sim.cfg.dt)

            #     force = wrench_sensor.data.force.torch[0, sensor_body_id]
            #     torque = wrench_sensor.data.torque.torch[0, sensor_body_id]

            #     if sim_time + 1e-12 >= next_ros:
            #         ros.publish(force, torque, frame_id)
            #         while next_ros <= sim_time:
            #             next_ros += ros_period
            #     ros.spin_some()

            #     if sim_time + 1e-12 >= next_print:
            #         print(
            #             f"[DROP] ep={episode_index:04d} | "
            #             f"F=[{float(force[0]):8.2f}, {float(force[1]):8.2f}, {float(force[2]):8.2f}] N | "
            #             f"T=[{float(torque[0]):8.2f}, {float(torque[1]):8.2f}, {float(torque[2]):8.2f}] N*m"
            #         )
            #         next_print += args_cli.print_period

            # -------------------------------------------------------------
            # Phase 2: put the object in front of the reset hand and sweep.
            # The object stays dynamic with gravity enabled.
            # -------------------------------------------------------------
            if not simulation_app.is_running():
                break

            hand_pos = robot.data.body_pos_w[0, sensor_body_id].detach().clone()
            collision_pos = hand_pos.clone()
            collision_pos[0] += args_cli.collision_offset_x
            collision_pos[2] += args_cli.collision_offset_z
            collision_pos[2] = max(
                float(collision_pos[2]),
                0.5 * args_cli.collision_size + 0.01,
            )

            collision_pose = torch.zeros(
                (args_cli.num_envs, 7), device=collision_object.device
            )
            collision_pose[:, 3] = 1.0
            collision_pose[0, :3] = collision_pos
            if args_cli.num_envs > 1:
                all_hand_pos = robot.data.body_pos_w[:, sensor_body_id].detach().clone()
                all_collision_pos = all_hand_pos.clone()
                all_collision_pos[:, 0] += args_cli.collision_offset_x
                all_collision_pos[:, 2] += args_cli.collision_offset_z
                all_collision_pos[:, 2] = torch.clamp(
                    all_collision_pos[:, 2],
                    min=0.5 * args_cli.collision_size + 0.01,
                )
                collision_pose[:, :3] = all_collision_pos

            collision_object.write_root_pose_to_sim(collision_pose)
            collision_object.write_root_velocity_to_sim(
                torch.zeros((args_cli.num_envs, 6), device=collision_object.device)
            )
            collision_object.write_data_to_sim()
            sim.step()
            sim_time += sim.cfg.dt
            scene.update(sim.cfg.dt)

            print(
                f"[PHASE] ep={episode_index} sweep | "
                f"CollisionObject=[{float(collision_pose[0,0]):.3f}, "
                f"{float(collision_pose[0,1]):.3f}, "
                f"{float(collision_pose[0,2]):.3f}] m"
            )

            # Number of physics steps consumed so far in this episode.
            episode_step = drop_steps
            # The code above decremented drop_steps to zero, so compute the
            # actual completed count from the fixed drop step count.
            completed_steps = min(
                int(round(args_cli.drop_duration / sim.cfg.dt)),
                args_cli.episode_steps,
            ) + 1
            sweep_step = max(completed_steps, 0)
            phase_start = sim_time

            while simulation_app.is_running() and sweep_step < args_cli.episode_steps:
                t = sim_time - phase_start
                target_q = default_q.clone()
                target_q[:, motion_joint_id] += (
                    args_cli.motion_amplitude
                    * math.sin(2.0 * math.pi * args_cli.motion_frequency * t)
                )

                robot.set_joint_position_target(target_q)
                scene.write_data_to_sim()
                sim.step()
                sim_time += sim.cfg.dt
                sweep_step += 1
                scene.update(sim.cfg.dt)

                force = wrench_sensor.data.force.torch[0, sensor_body_id]
                torque = wrench_sensor.data.torque.torch[0, sensor_body_id]

                if sim_time + 1e-12 >= next_ros:
                    ros.publish(force, torque, frame_id)
                    while next_ros <= sim_time:
                        next_ros += ros_period
                ros.spin_some()

                if sim_time + 1e-12 >= next_print:
                    print(
                        f"[SWEEP] ep={episode_index:04d} step={sweep_step:05d} | "
                        f"F=[{float(force[0]):9.2f}, {float(force[1]):9.2f}, {float(force[2]):9.2f}] N | "
                        f"T=[{float(torque[0]):9.2f}, {float(torque[1]):9.2f}, {float(torque[2]):9.2f}] N*m"
                    )
                    next_print += args_cli.print_period

            if not simulation_app.is_running():
                break

            # -------------------------------------------------------------
            # Fixed-step episode reset.
            # -------------------------------------------------------------
            episode_index += 1
            reset_episode()

            # Optional short reset pause. This is simulation time, not a
            # wall-clock sleep, so ROS continues to receive samples.
            for _ in range(max(args_cli.reset_pause_steps, 0)):
                if not simulation_app.is_running():
                    break
                robot.set_joint_position_target(default_q)
                scene.write_data_to_sim()
                sim.step()
                sim_time += sim.cfg.dt
                scene.update(sim.cfg.dt)

                force = wrench_sensor.data.force.torch[0, sensor_body_id]
                torque = wrench_sensor.data.torque.torch[0, sensor_body_id]
                if sim_time + 1e-12 >= next_ros:
                    ros.publish(force, torque, frame_id)
                    while next_ros <= sim_time:
                        next_ros += ros_period
                ros.spin_some()

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received.")
    finally:
        ros.shutdown()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
