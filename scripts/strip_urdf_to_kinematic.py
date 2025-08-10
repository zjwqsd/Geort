import sys
import xml.etree.ElementTree as ET
from pathlib import Path

def strip_urdf(src_path: Path, dst_path: Path):
    tree = ET.parse(src_path)
    root = tree.getroot()

    # 命名空间处理（URDF 通常无命名空间；这里做兼容）
    def _all(elem, tag):
        return list(elem.findall(tag)) + list(elem.findall(f".//{tag}"))

    # 1) 删除所有 <collision> 与 <inertial>
    removed_collision = removed_inertial = 0
    for link in root.findall(".//link"):
        # 移除 collision
        for coll in list(link.findall("collision")):
            link.remove(coll); removed_collision += 1
        # 移除 inertial
        for inert in list(link.findall("inertial")):
            link.remove(inert); removed_inertial += 1

    # 可选：也能顺带删掉 <dynamics>（若你完全不用摩擦/阻尼）
    for j in root.findall(".//joint"):
        for dyn in list(j.findall("dynamics")):
            j.remove(dyn)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_path, encoding="utf-8", xml_declaration=True)
    print(f"[OK] 写出：{dst_path}")
    print(f"移除 collision: {removed_collision} 个, inertial: {removed_inertial} 个")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python strip_urdf_to_kinematic.py <源URDF> [输出URDF]")
        sys.exit(1)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_kinematic.urdf")
    strip_urdf(src, dst)
