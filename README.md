# 📦 GeoRT（几何映射重定向）- 中文版增强说明

本仓库基于 [Meta 官方 GeoRT 实现](https://github.com/facebookresearch/GeoRT) 进行扩展，适配了自定义机械手、多任务数据采集与可视化分析，并升级至兼容新版 SAPIEN 与 Python 3.12 环境。

---

## ✅ 修改亮点

* 🧩 **兼容新版 SAPIEN 渲染引擎**
* ✂️ **支持手指关节范围裁剪 `joint_range_clip_ratio`**
* 👁️ **添加多种可视化与调试脚本**

> 📄 原始功能、训练流程、模型结构等请参考 [GeoRT 官方 README](https://github.com/facebookresearch/GeoRT)

---

## 🔧 安装与环境检查

```bash
conda create -n geort python=3.12
conda activate geort
pip install -r requirements.txt
pip install -e .
python testEnv.py  # 检查 torch, sapien 等依赖
```

---

## 🛠 使用流程概览（新增功能）

### ✅ Step 1：添加自定义机械手模型

1. 将你的 URDF 放入 `assets/your_hand_name/`
2. 创建对应配置文件 `geort/config/your_hand_name.json`

新增字段支持：

```json
"joint_range_clip_ratio": {
  "joint_name": [0.2, 0.9]
}
```

用于裁剪手指运动范围，避免不自然姿态。

可视化检查：

```bash
python geort/env/hand.py --hand xhand_right
```

---

### ✅ Step 2：采集人手关键点（Mediapipe 模拟 mocap）

使用 MediaPipe + RealSense：

```bash
python geort/mocap/mediapipe_mocap.py --name rot_alex
```

输出数据保存在 `data/rot_alex.npy`，用于训练。

---

### ✅ Step 3：训练模型

```bash
python geort/trainer.py -hand xhand_right -human_data rot_alex -ckpt_tag my_ckpt
```

---

### ✅ Step 4：部署推理

```python
import geort
model = geort.load_model('my_ckpt', epoch=-1)
qpos = model.forward(human_keypoints)
```

---

## 📊 可视化调试脚本（本仓库新增）

| 脚本路径                                    | 功能                |
| --------------------------------------- | ----------------- |
| `scripts/visualize_fk.py`               | FK 网络输出与实际关键点对比   |
| `scripts/visualize_human_data.py`        | 动画展示 21 点人手轨迹     |
| `scripts/visualize_robot_pointcloud.py` | Robot 手指点云可视化     |
| `scripts/visualize_human_pointcloud.py` | IK 训练样本可视化（点云+骨架） |

使用示例：

```bash
python scripts/visualize_fk.py --hand xhand_right
python scripts/visualize_human_data.py --hand rot_alex
```

---

## ⚙️ 配置模板简要说明

```json
{
  "name": "xhand_right",
  "urdf_path": "./assets/xhand_right/urdf/xhand_right.urdf",
  "base_link": "right_hand_link",
  "joint_order": [ ... ],
  "fingertip_link": [ ... ],
  "joint_range_clip_ratio": {
    "joint_1": [0.2, 0.8]
  }
}
```

详见：`geort/config/template.py`

---

## 🔗 参考与许可

* 原始项目地址：[facebookresearch/GeoRT](https://github.com/facebookresearch/GeoRT)
* 当前版本遵循 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 非商业许可协议
* 若使用本版本代码用于研究或教学，请注明来源

---

## 📬 联系

如需反馈或交流，请提交 issue 或联系本地维护者
