"""
Test CIFAR-10 data download and basic properties
"""

import torch
from torchvision import datasets, transforms
import os


def test_cifar10_download():
    """Test CIFAR-10 download and basic properties"""
    print("📥 Testing CIFAR-10 data download...")

    try:
        data_root = "./test_data"

        # Test download
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_dataset = datasets.CIFAR10(
            root=data_root, train=True, transform=transform, download=True
        )

        test_dataset = datasets.CIFAR10(
            root=data_root, train=False, transform=transform, download=True
        )

        # Check dataset sizes
        assert len(train_dataset) == 50000, f"Wrong train size: {len(train_dataset)}"
        assert len(test_dataset) == 10000, f"Wrong test size: {len(test_dataset)}"

        # Check sample
        sample_x, sample_y = train_dataset[0]
        assert sample_x.shape == (3, 32, 32), f"Wrong sample shape: {sample_x.shape}"
        assert 0 <= sample_y <= 9, f"Wrong label: {sample_y}"

        print(f"✅ CIFAR-10 downloaded successfully")
        print(f"   Train samples: {len(train_dataset):,}")
        print(f"   Test samples: {len(test_dataset):,}")
        print(f"   Sample shape: {sample_x.shape}")
        print(f"   Data location: {data_root}")

        # Cleanup
        import shutil

        if os.path.exists(data_root):
            shutil.rmtree(data_root)

        return True

    except Exception as e:
        print(f"❌ CIFAR-10 download failed: {e}")
        return False


if __name__ == "__main__":
    test_cifar10_download()
