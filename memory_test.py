"""
Memory usage test to ensure models fit in available memory
"""

import torch
import psutil
import os

from torch import nn
from Extended_Essay_Model import (
    FCNN_Small,
    FCNN_Medium,
    FCNN_Large,
    CNN_Small,
    CNN_Medium,
    CNN_Large,
    count_parameters,
)


def test_memory_usage():
    """Test memory requirements for all models"""
    print("🧠 Testing memory usage for all models...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    if device.type == "cuda":
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        )

    batch_size = 128  # Same as in main code
    input_tensor = torch.randn(batch_size, 3, 32, 32).to(device)

    models_to_test = [
        ("FCNN_Small", FCNN_Small),
        ("FCNN_Medium", FCNN_Medium),
        ("FCNN_Large", FCNN_Large),
        ("CNN_Small", CNN_Small),
        ("CNN_Medium", CNN_Medium),
        ("CNN_Large", CNN_Large),
    ]

    for model_name, model_class in models_to_test:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Get initial memory
            if device.type == "cuda":
                initial_gpu_mem = torch.cuda.memory_allocated() / (1024**2)
            initial_cpu_mem = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

            # Create model and optimizer
            model = model_class().to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.001, weight_decay=1e-4
            )
            criterion = nn.CrossEntropyLoss()

            # Forward pass
            model.train()
            outputs = model(input_tensor)
            loss = criterion(outputs, torch.randint(0, 10, (batch_size,)).to(device))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Measure peak memory
            if device.type == "cuda":
                peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024**2)
                gpu_usage = peak_gpu_mem - initial_gpu_mem
            else:
                gpu_usage = 0

            current_cpu_mem = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            cpu_usage = current_cpu_mem - initial_cpu_mem

            param_count = count_parameters(model)

            print(f"✅ {model_name}:")
            print(f"   Parameters: {param_count:,}")
            print(f"   GPU Memory: {gpu_usage:.1f} MB")
            print(f"   CPU Memory: {cpu_usage:.1f} MB")

            # Clean up
            del model, optimizer, outputs, loss

        except Exception as e:
            print(f"❌ {model_name}: Memory test failed - {e}")
            return False

    return True


if __name__ == "__main__":
    test_memory_usage()
