# from isaacsim import SimulationApp

# simulation_app = SimulationApp({
#     "headless": True
# })
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka cuMotion RMPFlow verification")
parser.add_argument("--usd", type=str,
                    default="/home/yh/tianji/mission_docs/mission01_v1/World0.usd")
parser.add_argument("--target-part", type=str,
                    default="/World/katao/katao",
                    choices=["/World/tighter_v1", "/World/part01"])
parser.add_argument("--offset", type=float, nargs=3,
                    default=(0.0, 0.0, 0.10),
                    metavar=("DX", "DY", "DZ"),
                    help="EE offset in target-part local frame, meters.")
parser.add_argument("--target-quat", type=float, nargs=4, default=None,
                    metavar=("QX", "QY", "QZ", "QW"))
parser.add_argument("--gripper", type=str, default="open",
                    choices=["open", "close"])
parser.add_argument("--print-every", type=int, default=50)
parser.add_argument("--target-tolerance", type=float, default=0.015)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.app

_extension_manager = omni.kit.app.get_app().get_extension_manager()
if not _extension_manager.is_extension_enabled("isaacsim.robot_motion.cumotion"):
    result = _extension_manager.set_extension_enabled_immediate(
        "isaacsim.robot_motion.cumotion", True
    )
    if not result:
        raise RuntimeError(
            "Could not enable isaacsim.robot_motion.cumotion"
        )

simulation_app.update()

print("=" * 60)

import isaacsim.robot_motion
print("isaacsim.robot_motion       OK")

import isaacsim.robot_motion.cumotion as cumotion
import os

print("isaacsim.robot_motion.cumotion OK")

from isaacsim.robot_motion.cumotion import RmpFlowController
print("RmpFlowController           OK")

print("=" * 60)

#########################################################################################################
from pathlib import Path
import re

urdf = Path(
    "/home/yh/miniforge3/envs/sim/lib/python3.12/site-packages/"
    "isaacsim/exts/isaacsim.robot_motion.cumotion/"
    "robot_configurations/franka/robot.urdf"
)

text = urdf.read_text()

# 打印包含这些 link 的 joint 定义
for parent, child in [
    ("panda_link7", "panda_hand"),
    ("panda_hand", "panda_leftfinger"),
    ("panda_hand", "panda_rightfinger"),
]:
    print(f"\n========== {parent} -> {child} ==========")

    pattern = (
        rf'<joint\b[^>]*>.*?'
        rf'<parent\s+link="{re.escape(parent)}"\s*/?>.*?'
        rf'<child\s+link="{re.escape(child)}"\s*/?>.*?'
        rf'</joint>'
    )

    matches = re.findall(pattern, text, re.DOTALL)

    for m in matches:
        print(m)

lines = text.splitlines()

for i in range(265, 325):
    print(f"{i+1:4d}: {lines[i]}")

# urdf = Path(
#     "/home/yh/miniforge3/envs/sim/lib/python3.12/site-packages/"
#     "isaacsim/exts/isaacsim.robot_motion.cumotion/"
#     "robot_configurations/franka/robot.urdf"
# )

# text = urdf.read_text()

# for i, line in enumerate(text.splitlines(), 1):
#     if (
#         'name="panda_hand"' in line
#         or 'name="panda_leftfinger"' in line
#         or 'name="panda_rightfinger"' in line
#         or 'name="panda_link7"' in line
#     ):
#         print(f"{i:4d}: {line}")

# cumotion_dir = Path(cumotion.__file__).parent

# print("cumotion_dir =", cumotion_dir)

# robot_dir = cumotion_dir / "robot_configurations" / "franka"

# print("franka_dir =", robot_dir)
# print("exists =", robot_dir.exists())

# if robot_dir.exists():
#     for p in sorted(robot_dir.rglob("*")):
#         if p.is_file():
#             print(p)

# franka_dir = Path(cumotion.__file__).parent / "robot_configurations" / "franka"

# for p in franka_dir.rglob("*"):
#     if p.is_file():
#         try:
#             text = p.read_text(errors="ignore")
#         except Exception:
#             continue

#         if "panda_leftfingertip" in text or "tool_frames" in text:
#             print("\n==========", p, "==========")
#             for i, line in enumerate(text.splitlines(), 1):
#                 if "panda_leftfingertip" in line or "tool_frames" in line:
#                     print(f"{i}: {line}")

# ext_dir = Path(
#     "/home/yh/miniforge3/envs/sim/lib/python3.12/site-packages/isaacsim/exts/"
#     "isaacsim.robot_motion.cumotion"
# )

# print("extension root:", ext_dir)
# print("exists:", ext_dir.exists())

# robot_config_dir = ext_dir / "robot_configurations"

# print("robot_configurations:", robot_config_dir)
# print("exists:", robot_config_dir.exists())

# if robot_config_dir.exists():
#     for p in sorted(robot_config_dir.rglob("*")):
#         if p.is_file():
#             print(p)

# xrdf = ext_dir / "robot_configurations" / "franka" / "robot.xrdf"

# print("XRDF:", xrdf)
# print("exists:", xrdf.exists())

# text = xrdf.read_text()

# for i, line in enumerate(text.splitlines(), 1):
#     if (
#         "tool" in line.lower()
#         or "panda_leftfingertip" in line
#         or "panda_hand" in line
#     ):
#         print(f"{i:4d}: {line}")

###########################################################################################################









simulation_app.close()