#!/usr/bin/env python3
"""
Isaac Lab 3.0 + ROS 2 + JointWrenchSensor minimal verification — v4 (sweep).

相比 v3 的改动：
  1) 机械臂初始位姿固定为“垂直向上”(up_q，默认全零关节角 = Franka 标准朝上位姿)。
  2) 碰撞物体初始位姿放在“离地面很近、且靠近机械臂”的位置。
     物体位置由仿真真实 FK 自标定：取机械臂弯到 down_q 时末端的位置，把它压到贴近
     地面高度，保证每次下扫都能扫到物体。
  3) 去除 v3 里“每次重置时把物体从高处抛下/掉落”的 Phase 1 动作。
  4) 机械臂周期性动作 = up_q -> down_q -> up_q 的正弦混合：从垂直向上弯下来扫过物体，
     再抬回，循环往复。每个下扫周期都会在末端产生一次 6 维力脉冲，便于观察曲线。
  5) 碰撞物体默认 kinematic（固定障碍），这样每圈扫过的力曲线干净、可重复；
     可通过 --collision_mode dynamic 改为可推动的动态物体。

两个 wrench 接口保持与 v3 一致：
  1) geometry_msgs/msg/WrenchStamped  ->  /franka/joint_wrench
  2) 六个 std_msgs/msg/Float64 标量话题 ->  /franka/joint_wrench/{fx,fy,fz,tx,ty,tz}
     （用 rqt_plot 同时画这 6 条即可观察末端 6 维力曲线）
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab JointWrenchSensor ROS2 demo (v4 sweep)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--physics_dt", type=float, default=1.0 / 120.0)
parser.add_argument("--ros_rate", type=float, default=100.0)
parser.add_argument("--topic", type=str, default="/franka/joint_wrench")
parser.add_argument("--body_name", type=str, default="panda_link7")

# --- v4 新增/改动的运动与物体参数 ---
parser.add_argument(
    "--up_q",
    type=str,
    default="0,0,0,-0.15,0,0,0",
    help="机械臂‘垂直向上’目标关节角(7个机械臂关节,逗号分隔,rad)。注意本资产 elbow(joint4)限位为负，"
         "故取 joint4≈-0.15 使手臂最接近竖直向上。",
)
parser.add_argument(
    "--down_q",
    type=str,
    default="0,-1.5,0,-2.7,0,1.5,0.785",
    help="机械臂‘弯下来’目标关节角(7个,逗号分隔)。下扫到该位姿时末端应贴近地面/物体。",
)
parser.add_argument("--sweep_frequency", type=float, default=0.25, help="下扫周期频率(Hz)，即 up->down->up 每秒次数")
parser.add_argument("--collision_size", type=float, default=0.20, help="碰撞物体边长(m)")
parser.add_argument("--collision_mass", type=float, default=5.0, help="碰撞物体质量(kg，仅 dynamic 模式生效)")
parser.add_argument(
    "--collision_mode",
    type=str,
    choices=["kinematic", "dynamic"],
    default="kinematic",
    help="kinematic=固定障碍(力曲线最干净可重复)；dynamic=可被推动的动态物体。",
)
parser.add_argument(
    "--collision_pos",
    type=str,
    default=None,
    help="可选：手动指定物体初始位姿 'x,y,z'(m)，覆盖 FK 自标定。",
)
parser.add_argument("--episode_steps", type=int, default=1200, help="每个 episode 的物理步数，到达后重置")
parser.add_argument("--reset_pause_steps", type=int, default=5, help="重置后保持的静止步数")
parser.add_argument("--print_period", type=float, default=0.25, help="终端打印周期(s)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _parse_arm_vec(s: str) -> list[float]:
    """解析 7 个机械臂关节角(不含夹爪)。夹爪关节由资产默认值补齐。"""
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 7:
        raise ValueError(f"up_q/down_q 需要 7 个机械臂关节角，但得到 {len(vals)} 个: {s!r}")
    return vals


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
    """Ground + Franka + one collision object (near ground, near robot)."""

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
                kinematic_enabled=(args_cli.collision_mode == "kinematic"),
                disable_gravity=(args_cli.collision_mode == "kinematic"),
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
                diffuse_color=(0.15, 0.55, 0.85),
                roughness=0.6,
            ),
        ),
        # 初始位姿在 reset_episode() 里由真实 FK 自标定写入，这里只是占位。
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, args_cli.collision_size / 2.0 + 0.01)),
    )

    joint_wrench = JointWrenchSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        update_period=0.0,
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

    num_joints = robot.data.default_joint_pos.shape[1]
    print("\n" + "=" * 80)
    print("[INFO] Franka bodies:")
    for i, name in enumerate(robot.body_names):
        print(f"  body[{i:02d}] = {name}")
    print("\n[INFO] Franka joints:")
    for i, name in enumerate(robot.joint_names):
        print(f"  joint[{i:02d}] = {name}")

    # 注意：JointWrenchSensor 没有 find_bodies 方法(那是 Articulation/RigidObject 的)，
    # 直接调用会抛 AttributeError 导致仿真被 finally 关闭。改用机械臂本体查询 body 索引。
    # 传感器挂载在整臂上、按机械臂 body 顺序跟踪全部 11 个 body，因此同一索引既可查
    # robot.data.body_pos_w，也可索引 wrench_sensor.data.force/torque，顺序一致。
    sensor_body_id = None
    try:
        ids, names = wrench_sensor.find_bodies(args_cli.body_name)
        if len(ids) == 1:
            sensor_body_id = int(ids[0])
    except Exception:
        sensor_body_id = None
    if sensor_body_id is None:
        ids, names = robot.find_bodies(args_cli.body_name)
        if len(ids) != 1:
            raise RuntimeError(
                f"Body lookup failed for '{args_cli.body_name}': names={names}, ids={ids}"
            )
        sensor_body_id = int(ids[0])
    print(f"[INFO] Selected wrench body: {args_cli.body_name} (id={sensor_body_id})")

    # ---- 解析 up/down 机械臂关节角，并补齐到资产实际的 9 维(含 2 个夹爪关节) ----
    # Franka 资产共有 9 个关节：panda_joint1..7 + panda_finger_joint1/2。
    # 我们只描述 7 个手臂关节，夹爪关节沿用资产默认值(张开)。
    num_arm = 7
    up_arm = _parse_arm_vec(args_cli.up_q)
    down_arm = _parse_arm_vec(args_cli.down_q)
    if len(up_arm) != num_arm or len(down_arm) != num_arm:
        raise RuntimeError(
            f"up_q/down_q 必须是 {num_arm} 个机械臂关节角，得到 {len(up_arm)}/{len(down_arm)}"
        )
    # default_q_full: (num_envs, num_joints)，含合法默认夹爪角，可直接写回仿真。
    default_q_full = robot.data.default_joint_pos.torch.clone()
    up_q = default_q_full.clone()
    down_q = default_q_full.clone()
    up_q[:, :num_arm] = torch.tensor(up_arm, dtype=up_q.dtype, device=up_q.device)
    down_q[:, :num_arm] = torch.tensor(down_arm, dtype=down_q.dtype, device=down_q.device)
    zero_qd = torch.zeros_like(default_q_full)

    print("[INFO] Wrench convention: incoming_joint_frame")
    print(f"[INFO] up_q   (垂直向上, 手臂) = {[round(float(x),3) for x in up_arm]}")
    print(f"[INFO] down_q (弯下来, 手臂)   = {[round(float(x),3) for x in down_arm]}")
    print(f"[INFO] sweep_frequency   = {args_cli.sweep_frequency} Hz")
    print(f"[INFO] collision_mode    = {args_cli.collision_mode}")
    print(f"[INFO] Episode reset period: {args_cli.episode_steps} physics steps")
    print("[INFO] No drop/throw phase (v4). Object placed near ground near robot.")
    print("=" * 80 + "\n")

    ros = WrenchRosPublisher(args_cli.topic)
    frame_id = f"{args_cli.body_name}_incoming_joint"

    ros_period = 1.0 / max(args_cli.ros_rate, 1e-6)
    next_ros = 0.0
    next_print = 0.0
    sim_time = 0.0
    episode_index = 0

    def reset_episode() -> None:
        nonlocal next_print

        scene.reset()

        # 先把机械臂摆到 “垂直向上” 位姿，读取末端位置。
        robot.write_joint_state_to_sim(up_q, zero_qd)
        robot.set_joint_position_target(up_q)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.cfg.dt)
        hand_up = robot.data.body_pos_w[0, sensor_body_id].detach().clone()

        # 再把机械臂摆到 “弯下来” 位姿，用真实 FK 读取末端位置，用来摆放物体。
        robot.write_joint_state_to_sim(down_q, zero_qd)
        robot.set_joint_position_target(down_q)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.cfg.dt)
        hand_down = robot.data.body_pos_w[0, sensor_body_id].detach().clone()

        # 复位回 “垂直向上”，作为 episode 的起点（垂直向上 -> 下扫）。
        robot.write_joint_state_to_sim(up_q, zero_qd)
        robot.set_joint_position_target(up_q)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.cfg.dt)

        # ---- 摆放碰撞物体：贴近地面 + 靠近机械臂（弯下来时末端所在处） ----
        rest_z = args_cli.collision_size / 2.0 + 0.005
        near_ground_thresh = rest_z + 0.10  # 若末端下扫点足够低，就把物体压到贴地
        if args_cli.collision_pos is not None:
            ox, oy, oz = [float(x) for x in args_cli.collision_pos.split(",")]
        else:
            ox = float(hand_down[0])
            oy = float(hand_down[1])
            oz = rest_z if float(hand_down[2]) <= near_ground_thresh else float(hand_down[2])

        collision_pose = torch.zeros((args_cli.num_envs, 7), device=collision_object.device)
        collision_pose[:, 3] = 1.0  # 单位四元数
        collision_pose[0, :3] = torch.tensor([ox, oy, oz], device=collision_object.device)
        if args_cli.num_envs > 1:
            all_hand_down = robot.data.body_pos_w[:, sensor_body_id].detach().clone()
            collision_pose[:, 0] = all_hand_down[:, 0]
            collision_pose[:, 1] = all_hand_down[:, 1]
            for e in range(args_cli.num_envs):
                hz = float(all_hand_down[e, 2])
                collision_pose[e, 2] = rest_z if hz <= near_ground_thresh else hz

        collision_object.write_root_pose_to_sim(collision_pose)
        collision_object.write_root_velocity_to_sim(
            torch.zeros((args_cli.num_envs, 6), device=collision_object.device)
        )
        collision_object.write_data_to_sim()
        sim.step()
        scene.update(sim.cfg.dt)

        print(
            f"\n[RESET] episode={episode_index} | "
            f"hand_up_z={float(hand_up[2]):.3f} m | "
            f"hand_down=[{float(hand_down[0]):.3f},{float(hand_down[1]):.3f},{float(hand_down[2]):.3f}] m | "
            f"object=[{ox:.3f},{oy:.3f},{oz:.3f}] m"
        )
        if float(hand_down[2]) > near_ground_thresh:
            print(
                "  [WARN] 弯下来时末端离地偏高(>{:.2f}m)，物体未贴地。".format(near_ground_thresh)
                + " 可增大 --down_q 的弯腰/折肘幅度(更负)，让末端扫到更低处。"
            )
        next_print = sim_time

    try:
        reset_episode()

        while simulation_app.is_running():
            # -------------------------------------------------------------
            # 单一循环：up -> down -> up 正弦混合，周期性下扫扫过物体。
            #  t=0   : s=0   -> 垂直向上
            #  t=T/2 : s=1   -> 弯下来（末端扫到物体）
            # 每个周期产生一次 6 维力脉冲，可用 rqt_plot 观察曲线。
            # -------------------------------------------------------------
            phase_start = sim_time
            episode_step = 0

            while simulation_app.is_running() and episode_step < args_cli.episode_steps:
                t = sim_time - phase_start
                s = 0.5 - 0.5 * math.cos(2.0 * math.pi * args_cli.sweep_frequency * t)
                target_q = up_q + (down_q - up_q) * s

                robot.set_joint_position_target(target_q)
                scene.write_data_to_sim()
                sim.step()
                sim_time += sim.cfg.dt
                episode_step += 1
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
                        f"[SWEEP] ep={episode_index:04d} step={episode_step:05d} s={s:5.2f} | "
                        f"F=[{float(force[0]):9.2f}, {float(force[1]):9.2f}, {float(force[2]):9.2f}] N | "
                        f"T=[{float(torque[0]):9.2f}, {float(torque[1]):9.2f}, {float(torque[2]):9.2f}] N*m"
                    )
                    next_print += args_cli.print_period

            if not simulation_app.is_running():
                break

            # -------------------------------------------------------------
            # 固定步数后重置（不再抛/掉落物体，直接重新摆放）。
            # -------------------------------------------------------------
            episode_index += 1
            reset_episode()

            for _ in range(max(args_cli.reset_pause_steps, 0)):
                if not simulation_app.is_running():
                    break
                robot.set_joint_position_target(up_q)
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
