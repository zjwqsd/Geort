import importlib
import sys

def check_module(module_name, version_attr="__version__"):
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, version_attr, None)
        if version:
            print(f"✅ {module_name:<12} | version: {version}")
        else:
            print(f"✅ {module_name:<12} | (version not found)")
    except ImportError:
        print(f"❌ {module_name:<12} | not installed")

def check_torch():
    try:
        import torch
        print(f"✅ torch        | version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device       : {torch.cuda.get_device_name(0)}")
            print(f"  Arch list    : {torch.cuda.get_arch_list()}")
    except ImportError:
        print("❌ torch        | not installed")

def main():
    print("🔍 Checking Python version...")
    print(f"Python: {sys.version}\n")

    print("🔍 Checking installed packages...\n")

    # Special handling for torch
    check_torch()
    print()

    # General modules
    modules = [
        "torchvision",
        "torchaudio",
        "numpy",
        "scipy",
        "sapien",
        "tqdm",
        "open3d",
        "trimesh",
        "zmq"
    ]

    for m in modules:
        check_module(m)

    print("\n✅ Environment check complete.")

if __name__ == "__main__":
    main()
