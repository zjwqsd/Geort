import importlib
import sys

def check_module(module_name, alias=None):
    try:
        module = importlib.import_module(module_name)
        print(f"✅ {module_name} loaded", end='')
        if alias:
            version = getattr(module, alias, None)
            if version:
                print(f" | version: {version}")
            else:
                print()
        else:
            print()
    except ImportError:
        print(f"❌ {module_name} not installed")

def main():
    print("🔍 Checking Python version...")
    print(f"Python: {sys.version}\n")

    print("🔍 Checking installed packages...\n")

    check_module('torch')
    import torch
    print(f"  torch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Supported arch: {torch.cuda.get_arch_list()}")
    print()

    check_module('torchvision')
    check_module('torchaudio')
    check_module('numpy')
    import numpy
    print(f"  numpy version: {numpy.__version__}\n")

    check_module('scipy')
    check_module('sapien')
    check_module('gym')
    check_module('tqdm')
    check_module('open3d')
    check_module('trimesh')
    check_module('zmq')

    print("\n✅ Environment check complete.")

if __name__ == "__main__":
    main()
