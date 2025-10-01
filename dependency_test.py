"""
Test all required dependencies are installed
"""


def test_dependencies():
    """Test that all required packages are available"""
    print("📦 Testing dependencies...")

    required_packages = [
        "torch",
        "torchvision",
        "sklearn",
        "matplotlib",
        "numpy",
        "pandas",
        "psutil",
        "ptflops",
        "seaborn",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT FOUND")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All dependencies available!")
        return True


def test_torch_setup():
    """Test PyTorch setup"""
    print("\n🔥 Testing PyTorch setup...")

    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)")

    # Test basic tensor operations
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = torch.mm(x, y)

    print("✅ Basic tensor operations working")

    return True


if __name__ == "__main__":
    deps_ok = test_dependencies()
    torch_ok = test_torch_setup()

    if deps_ok and torch_ok:
        print("\n🎉 All dependencies ready!")
    else:
        print("\n❌ Please fix dependency issues before proceeding")
