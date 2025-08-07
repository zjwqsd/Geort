import sapien.core as sapien
import json
from geort.utils.config_utils import get_config
from geort.utils.hand_utils import get_entity_by_name, get_active_joints

def check_hand_config(hand_name: str):
    config = get_config(hand_name)
    urdf_path = config["urdf_path"]
    joint_order = config["joint_order"]
    fingertips = config.get("fingertip_link", [])
    base_link = config["base_link"]

    print(f"\n🔍 加载 URDF: {urdf_path}")
    engine = sapien.Engine()
    scene = engine.create_scene()
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    hand = loader.load(urdf_path)

    all_links = hand.get_links()
    all_link_names = [link.get_name() for link in all_links]

    all_joints = get_active_joints(hand, joint_order)
    all_joint_names = [joint.get_name() for joint in all_joints]

    print("\n📌 URDF中所有 active joints:")
    for jn in all_joint_names:
        print(f"- {jn}")

    print("\n📌 URDF中所有 link 名称:")
    for ln in all_link_names:
        print(f"- {ln}")

    print("\n✅ 检查 joint_order 是否在 URDF 中:")
    for jn in joint_order:
        if jn in all_joint_names:
            print(f"  ✅ {jn}")
        else:
            print(f"  ❌ 不存在于 URDF 中: {jn}")

    print("\n✅ 检查 fingertip_link 是否在 URDF 中:")
    for tip in fingertips:
        link_name = tip["link"]
        if link_name in all_link_names:
            print(f"  ✅ {link_name}")
        else:
            print(f"  ❌ 不存在于 URDF 中: {link_name}")

    print("\n✅ 检查每个手指的 joint 是否在 joint_order 中:")
    for tip in fingertips:
        finger = tip["name"]
        for joint_name in tip["joint"]:
            if joint_name in joint_order:
                print(f"  ✅ {finger}: {joint_name}")
            else:
                print(f"  ❌ {finger}: {joint_name} 不在 joint_order 中！")

    print("\n✅ 检查 base_link 是否存在:")
    if base_link in all_link_names:
        print(f"  ✅ {base_link}")
    else:
        print(f"  ❌ {base_link} 不存在于 URDF 中！")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", type=str, required=True, help="配置名称，例如 xhand_right")
    args = parser.parse_args()

    check_hand_config(args.hand)
