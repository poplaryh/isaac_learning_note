#!/usr/bin/env python3
"""Minimal Isaac Lab 3.0 + ROS 2 JointWrenchSensor demo.

Purpose:
  1. Open an existing USD scene that already contains a Franka articulation.
  2. Let Isaac Lab attach to the existing articulation (spawn=None).
  3. Create a JointWrenchSensor on that articulation.
  4. Read the incoming joint reaction wrench of one selected body.
  5. Publish it as geometry_msgs/msg/WrenchStamped over ROS 2.

Important:
  - JointWrenchSensor reports an incoming joint reaction wrench in the
    child-side incoming-joint frame, with torque referenced at the child-side
    joint anchor. It is NOT automatically a physical end-effector F/T sensor.
  - This demo publishes that raw sensor frame and names frame_id accordingly.

Example:
  ./isaaclab.sh -p isaaclab_joint_wrench_ros2_minimal.py \
      --scene_usd /absolute/path/to/assembly_scene.usd \
      --robot_prim /World/Robot/Franka \
      --body_name panda_link7 \
      --topic /franka/joint_wrench \
      --ros_rate 100

Then, in another terminal:
  ros2 topic echo /franka/joint_wrench

For curves, use PlotJuggler or rqt_plot.
"""

import argparse
import time

# Launch Isaac Sim before importing modules that depend on Kit/USD bindings.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab JointWrenchSensor -> ROS 2 demo")
parser.add_argument("--scene_usd", type=str, required=True, help="Existing USD scene containing the Franka articulation.")
parser.add_argument("--robot_prim", type=str, default="/World/Robot/Franka", help="Existing Franka articulation prim path.")
parser.add_argument("--body_name", type=str, default="panda_link7", help="Body whose incoming joint wrench will be published.")
parser.add_argument("--topic", type=str, default="/franka/joint_wrench", help="ROS 2 WrenchStamped topic.")
parser.add_argument("--frame_id", type=str, default="panda_link7_incoming_joint", help="Frame name used in WrenchStamped.header.frame_id.")
parser.add_argument("--ros_rate", type=float, default=100.0, help="ROS 2 publish rate in Hz. The sensor itself is sampled by the simulation loop.")
parser.add_argument("--physics_dt", type=float, default=1.0 / 120.0, help="Physics time step in seconds.")
parser.add_argument("--print_every", type=int, default=60, help="Print wrench to terminal every N simulation steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Everything below this line can safely import Isaac/ROS dependencies.
import rclpy
from geometry_msgs.msg import WrenchStamped
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.sensors import JointWrenchSensor, JointWrenchSensorCfg

# Official Isaac Lab asset package. This provides the actuator/joint metadata
# used to attach an Articulation interface to an already-existing Franka USD.
from isaaclab_assets import FRANKA_PANDA_CFG


class WrenchPublisher:
    """Small ROS 2 publisher wrapper. Publishing is called synchronously
    from the simulation loop so ROS does not drive physics stepping."""

    def __init__(self, topic: str, frame_id: str):
        self.node = rclpy.create_node("isaaclab_joint_wrench_publisher")
        self.publisher = self.node.create_publisher(WrenchStamped, topic, 10)
        self.frame_id = frame_id

    def publish(self, force_xyz, torque_xyz) -> None:
        msg = WrenchStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.wrench.force.x = float(force_xyz[0])
        msg.wrench.force.y = float(force_xyz[1])
        msg.wrench.force.z = float(force_xyz[2])
        msg.wrench.torque.x = float(torque_xyz[0])
        msg.wrench.torque.y = float(torque_xyz[1])
        msg.wrench.torque.z = float(torque_xyz[2])

        self.publisher.publish(msg)

    def spin_some(self) -> None:
        # Non-blocking ROS callback processing. We deliberately do not use
        # rclpy.spin(), because the simulation loop must remain synchronous.
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self) -> None:
        self.node.destroy_node()


def main() -> None:
    # ---------------------------------------------------------------------
    # 1) Initialize ROS 2.
    # ---------------------------------------------------------------------
    rclpy.init()
    ros_pub = WrenchPublisher(args_cli.topic, args_cli.frame_id)

    try:
        # -----------------------------------------------------------------
        # 2) Open the existing USD stage.
        # -----------------------------------------------------------------
        # The USD is expected to already contain the Franka and the rest of
        # your scene. We do NOT spawn another Franka here.
        opened = sim_utils.open_stage(args_cli.scene_usd)
        if opened is False:
            raise RuntimeError(f"Failed to open USD stage: {args_cli.scene_usd}")

        # -----------------------------------------------------------------
        # 3) Create the simulation context on the already-open stage.
        # -----------------------------------------------------------------
        sim_cfg = sim_utils.SimulationCfg(
            dt=args_cli.physics_dt,
            device=args_cli.device,
            create_stage_in_memory=False,
        )
        sim = SimulationContext(sim_cfg)

        # Keep the default viewer useful. Change this to your desired camera.
        if app_launcher.has_gui():
            sim.set_camera_view(
                eye=[1.5, 1.5, 1.2],
                target=[0.5, 0.0, 0.4],
            )

        # -----------------------------------------------------------------
        # 4) Attach Isaac Lab's Articulation interface to the existing Franka.
        # -----------------------------------------------------------------
        # spawn=None means the prim must already exist in the USD stage.
        # We reuse the official Franka configuration for joint/body ordering,
        # actuator metadata, etc., but replace the prim path and disable spawn.
        robot_cfg = FRANKA_PANDA_CFG.replace(
            prim_path=args_cli.robot_prim,
            spawn=None,
        )
        robot = Articulation(robot_cfg)

        # -----------------------------------------------------------------
        # 5) Create the JointWrenchSensor on the same articulation.
        # -----------------------------------------------------------------
        wrench_cfg = JointWrenchSensorCfg(
            prim_path=args_cli.robot_prim,
            update_period=0.0,
            debug_vis=False,
        )
        wrench_sensor = JointWrenchSensor(wrench_cfg)

        # Initialize PhysX handles / buffers.
        sim.reset()
        robot.reset()
        wrench_sensor.reset()

        # One update before reading names/data is useful after initialization.
        sim_dt = sim.get_physics_dt()
        robot.update(sim_dt)
        wrench_sensor.update(sim_dt)

        # Resolve the requested body name to the sensor's body index.
        body_ids, body_names = wrench_sensor.find_bodies([args_cli.body_name])
        if len(body_ids) != 1:
            raise RuntimeError(
                f"Could not uniquely resolve body '{args_cli.body_name}'. "
                f"Matched ids={body_ids}, names={body_names}."
            )
        body_id = int(body_ids[0])

        print("\n[INFO] Setup complete")
        print(f"[INFO] USD          : {args_cli.scene_usd}")
        print(f"[INFO] Robot prim   : {args_cli.robot_prim}")
        print(f"[INFO] Wrench body  : {body_names[0]}")
        print(f"[INFO] ROS topic    : {args_cli.topic}")
        print(f"[INFO] ROS frame_id : {args_cli.frame_id}")
        print(f"[INFO] Physics dt   : {sim_dt:.9f} s ({1.0 / sim_dt:.1f} Hz)")
        print(f"[INFO] ROS rate     : {args_cli.ros_rate:.1f} Hz")
        print("[INFO] Press Ctrl+C or close the Isaac Sim window to stop.\n")

        # -----------------------------------------------------------------
        # 6) Main loop.
        # -----------------------------------------------------------------
        step_count = 0
        next_ros_publish_time = time.perf_counter()
        ros_period = 1.0 / args_cli.ros_rate

        while simulation_app.is_running():
            # No control command is required for this sensor-only test.
            # The Franka simply remains in its current state.

            # Advance physics by exactly one physics step.
            sim.step()

            # Refresh Isaac Lab buffers after the step.
            robot.update(sim_dt)
            wrench_sensor.update(sim_dt)

            # Read the latest incoming joint reaction wrench.
            # Shape: (num_envs, num_bodies, 3). We use env 0 only.
            force = wrench_sensor.data.force.torch[0, body_id]
            torque = wrench_sensor.data.torque.torch[0, body_id]

            # Publish at the requested ROS rate. ROS does not control the
            # simulation clock; it only samples the latest simulated wrench.
            now = time.perf_counter()
            if now >= next_ros_publish_time:
                ros_pub.publish(force.detach().cpu().tolist(), torque.detach().cpu().tolist())
                next_ros_publish_time = now + ros_period

            # Process any ROS callbacks without blocking physics.
            ros_pub.spin_some()

            if args_cli.print_every > 0 and step_count % args_cli.print_every == 0:
                f = force.detach().cpu().tolist()
                t = torque.detach().cpu().tolist()
                print(
                    f"step={step_count:7d}  "
                    f"F=[{f[0]: .4f}, {f[1]: .4f}, {f[2]: .4f}] N  "
                    f"T=[{t[0]: .4f}, {t[1]: .4f}, {t[2]: .4f}] N.m"
                )

            step_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        ros_pub.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
