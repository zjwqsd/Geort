import pybullet as p
import pybullet_data
import os

# 设置路径（替换为你的 URDF 路径）
urdf_path = r"assets\xhand_right\urdf\xhand_right.urdf"

# 启动 PyBullet 物理引擎
p.connect(p.DIRECT)  # GUI: p.GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())

try:
    robot = p.loadURDF(urdf_path, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
    print("✅ URDF 加载成功，未检测到惯性矩阵报错。")
except Exception as e:
    print("❌ 加载 URDF 失败：")
    print(e)
