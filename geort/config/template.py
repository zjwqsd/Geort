allegro_hand_config = {
    "name": "allegro_right",   # 手型名称，建议使用小写字母 + 下划线风格，如 allegro_right、xhand_right

    # URDF 文件路径（相对于项目根目录）
    "urdf_path": "./assets/allegro_right/allegro_hand_right.urdf",

    # 基准链接（base_link）名称
    # GeoRT 中所有的关键点坐标都将被转换到这个链接的坐标系下
    # 建议设为掌心固定的关节或添加虚拟链接使其满足以下朝向规范：
    # - Y 轴指向拇指
    # - Z 轴指向中指
    # - X 轴垂直于手掌朝外
    "base_link": "base_link",

    # 关节顺序（joint_order）
    # GeoRT 所有模型输入输出的关节顺序必须严格按照这里定义的顺序排列
    # 建议与控制器接口中的关节顺序保持一致
    "joint_order": [
        "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",
        "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",
        "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",
        "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"
    ],

    # 指尖链接定义列表（fingertip_link）
    # 每个元素代表一根手指的末端，包含其对应的：
    # - 名称（name）：用于调试和可视化；
    # - 末端链接名称（link）：URDF 中代表指尖的 link；
    # - 控制该指尖的所有关节名（joint）；
    # - 相对该 link 的指尖中心偏移（center_offset）；
    # - 对应的人手关键点 ID（human_hand_id）：与 MediaPipe 的 21 关键点对齐
    "fingertip_link": [
        {
            "name": "index",              # 食指
            "link": "link_3.0_tip",
            "joint": ["joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0"],
            "center_offset": [0.0, 0.0, -0.005],
            "human_hand_id": 8
        },
        {
            "name": "middle",             # 中指
            "link": "link_7.0_tip",
            "joint": ["joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0"],
            "center_offset": [0.0, 0.0, -0.005],
            "human_hand_id": 12
        },
        {
            "name": "ring",               # 无名指
            "link": "link_11.0_tip",
            "joint": ["joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0"],
            "center_offset": [0.0, 0.0, -0.005],
            "human_hand_id": 16
        },
        {
            "name": "thumb",              # 拇指
            "link": "link_15.0_tip",
            "joint": ["joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"],
            "center_offset": [0.0, 0.0, -0.005],
            "human_hand_id": 4
        }
    ],

    # 可选字段：关节运动范围裁剪比例（joint_range_clip_ratio）
    # 若指定某个关节的 [a, b]，则其 joint_limit 将变为：
    #   clipped_min = urdf_min + (urdf_max - urdf_min) * a
    #   clipped_max = urdf_min + (urdf_max - urdf_min) * b
    # 这样可避免某些不自然或无法控制的极限位置，使模型训练更加稳定。
    "joint_range_clip_ratio": {
        "joint_0.0": [0.1, 0.9],
        "joint_12.0": [0.3, 1.0]
    }
}
