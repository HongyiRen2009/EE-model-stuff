"""
Quick tests to validate model architectures and basic functionality
"""

import torch
import torch.nn as nn
import time
from Extended_Essay_Model import *


def test_model_architectures():
    """Test that all models can be instantiated and perform forward pass"""
    print("🔧 Testing model architectures...")

    device = torch.device("cpu")  # Use CPU for quick tests
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 32, 32)

    # All model classes to test
    models_to_test = [
        ("FCNN_Small", FCNN_Small),
        ("FCNN_Medium", FCNN_Medium),
        ("FCNN_Large", FCNN_Large),
        ("FCNN_Small_No_Reg", FCNN_Small_No_Reg),
        ("FCNN_Medium_No_Reg", FCNN_Medium_No_Reg),
        ("FCNN_Large_No_Reg", FCNN_Large_No_Reg),
        ("CNN_Small", CNN_Small),
        ("CNN_Medium", CNN_Medium),
        ("CNN_Large", CNN_Large),
        ("CNN_Small_No_Reg", CNN_Small_No_Reg),
        ("CNN_Medium_No_Reg", CNN_Medium_No_Reg),
        ("CNN_Large_No_Reg", CNN_Large_No_Reg),
    ]

    for model_name, model_class in models_to_test:
        try:
            # Test instantiation
            model = model_class().to(device)

            # Test forward pass
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)

            # Check output shape
            assert output.shape == (
                batch_size,
                10,
            ), f"{model_name}: Wrong output shape {output.shape}"

            # Check parameter count
            param_count = count_parameters(model)
            assert param_count > 0, f"{model_name}: No parameters found"

            print(f"✅ {model_name}: {param_count:,} parameters")

        except Exception as e:
            print(f"❌ {model_name}: Failed - {e}")
            return False

    return True


def test_data_loaders():
    """Test data loading functionality"""
    print("\n🔧 Testing data loaders...")

    try:
        # Test with minimal transforms
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        # Small batch size for quick test
        train_loader, val_loader, test_loader = get_data_loaders(
            transform_train, transform_test, batch_size=8, val_split=0.1
        )

        # Test one batch from each loader
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))

        # Check shapes
        for name, (x, y) in [
            ("train", train_batch),
            ("val", val_batch),
            ("test", test_batch),
        ]:
            assert x.shape[1:] == (3, 32, 32), f"{name}: Wrong input shape {x.shape}"
            assert len(y) == x.shape[0], f"{name}: Mismatched batch size"
            assert torch.all((y >= 0) & (y <= 9)), f"{name}: Invalid labels"

        print(
            f"✅ Data loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)} batches"
        )
        return True

    except Exception as e:
        print(f"❌ Data loaders failed: {e}")
        return False


def test_utility_functions():
    """Test utility functions"""
    print("\n🔧 Testing utility functions...")

    try:
        # Test reproducibility
        ensure_reproducibility(42)

        # Test parameter counting
        model = FCNN_Small()
        param_count = count_parameters(model)
        assert param_count > 0

        # Test FLOPS computation
        macs, params = compute_flops_params(model)
        assert macs > 0 and params > 0

        print(f"✅ Utility functions: {param_count:,} params, {macs:,} MACs")
        return True

    except Exception as e:
        print(f"❌ Utility functions failed: {e}")
        return False


def test_training_loop_short():
    """Test training loop with minimal epochs"""
    print("\n🔧 Testing training loop (1 epoch)...")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get small data loaders
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_loader, val_loader, test_loader = get_data_loaders(
            transform_train, transform_test, batch_size=16, val_split=0.1
        )

        # Use small model
        model = FCNN_Small().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Train for just 1 epoch
        save_dir = "test_results"
        os.makedirs(save_dir, exist_ok=True)

        model, results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=1,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            save_dir=save_dir,
        )

        # Check results structure
        required_keys = [
            "final_test_acc",
            "final_test_loss",
            "best_val_acc",
            "training_time",
            "parameters",
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"

        print(f"✅ Training loop: Test acc = {results['final_test_acc']:.4f}")

        # Cleanup with proper Windows permissions handling
        if os.path.exists(save_dir):
            import shutil
            import stat

            def handle_remove_readonly(func, path, exc):
                """Error handler for Windows readonly files"""
                if os.path.exists(path):
                    os.chmod(path, stat.S_IWRITE)
                    func(path)

            try:
                shutil.rmtree(save_dir, onerror=handle_remove_readonly)
            except PermissionError:
                print(f"⚠️  Could not remove {save_dir} - please delete manually")

        return True

    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        return False


def test_model_variants_config():
    """Test model variants configuration"""
    print("\n🔧 Testing model variants configuration...")

    try:
        # Check that all model variants are valid
        for model_name, model_info in model_variants.items():
            model_class = model_info["model"]
            augmentation = model_info["augmentation"]

            # Check model can be instantiated
            model = model_class()
            assert hasattr(model, "forward"), f"{model_name}: No forward method"

            # Check augmentation exists
            assert (
                augmentation in augmentations
            ), f"{model_name}: Invalid augmentation {augmentation}"

        print(f"✅ Model variants: {len(model_variants)} configurations valid")
        return True

    except Exception as e:
        print(f"❌ Model variants failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("🧪 Running quick validation tests...\n")

    tests = [
        ("Model Architectures", test_model_architectures),
        ("Data Loaders", test_data_loaders),
        ("Utility Functions", test_utility_functions),
        ("Model Variants Config", test_model_variants_config),
        ("Training Loop", test_training_loop_short),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")

        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")

    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print(f"{'='*60}")

    if passed != total:
        print("⚠️  Some tests failed. Please fix before deploying.")
        return False
    return True


if __name__ == "__main__":
    # Set deterministic behavior for testing
    torch.manual_seed(42)
    # Run tests
    success = run_all_tests()

    if success:
        print("\n📊 Quick parameter count check:")
        getParameterCounts()
