import xml.etree.ElementTree as ET
import os

URDF_PATH = r"assets\xhand_right\urdf\xhand_right.urdf"
FIXED_URDF_PATH = URDF_PATH.replace(".urdf", "_fixed.urdf")

bad_links = [
    "right_hand_ee_link",
    "right_hand_back_link",
    "right_hand_thumb_rotaback_link1",
    "right_hand_thumb_rotaback_link2",
    "right_hand_thumb_rota_tip",
    "right_hand_index_rotaback_link1",
    "right_hand_index_rotaback_link2",
    "right_hand_index_rota_tip",
    "right_hand_midback_link1",
    "right_hand_midback_link2",
    "right_hand_mid_tip",
    "right_hand_ringback_link1",
    "right_hand_ringback_link2",
    "right_hand_ring_tip",
    "right_hand_pinkyback_link1",
    "right_hand_pinkyback_link2",
    "right_hand_pinky_tip",
]

def create_default_inertial(ns=""):
    inertial = ET.Element(f"{ns}inertial")

    origin = ET.SubElement(inertial, f"{ns}origin")
    origin.set("xyz", "0 0 0")
    origin.set("rpy", "0 0 0")

    mass = ET.SubElement(inertial, f"{ns}mass")
    mass.set("value", "0.01")

    inertia = ET.SubElement(inertial, f"{ns}inertia")
    inertia.set("ixx", "1e-5")
    inertia.set("iyy", "1e-5")
    inertia.set("izz", "1e-5")
    inertia.set("ixy", "0")
    inertia.set("ixz", "0")
    inertia.set("iyz", "0")

    return inertial

def is_invalid_inertial(inertial, ns=""):
    if inertial is None:
        return True
    mass_tag = inertial.find(f"{ns}mass")
    inertia_tag = inertial.find(f"{ns}inertia")
    if mass_tag is None or inertia_tag is None:
        return True
    try:
        mass = float(mass_tag.get("value", "0"))
        ixx = float(inertia_tag.get("ixx", "0"))
        iyy = float(inertia_tag.get("iyy", "0"))
        izz = float(inertia_tag.get("izz", "0"))
        epsilon = 1e-6
        return (
            mass < epsilon or
            ixx < epsilon or
            iyy < epsilon or
            izz < epsilon
        )
    except Exception:
        return True

def fix_urdf_inertial():
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()

    # 检查是否有命名空间
    ns = ''
    if '}' in root.tag:
        ns = root.tag.split('}')[0] + '}'

    modified = 0
    for link in root.findall(f"{ns}link"):
        name = link.attrib.get("name")
        if name not in bad_links:
            continue

        inertial = link.find(f"{ns}inertial")
        if is_invalid_inertial(inertial, ns):
            if inertial is not None:
                link.remove(inertial)
            link.append(create_default_inertial(ns))
            print(f"✅ 修复 link: {name}")
            modified += 1

    tree.write(FIXED_URDF_PATH)
    print(f"\n💾 修复完成，共修复 {modified} 个 link。已保存为：{FIXED_URDF_PATH}")

if __name__ == "__main__":
    fix_urdf_inertial()
