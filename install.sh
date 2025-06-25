#!/bin/bash
set -e  # Exit on error

echo "🚀 Starting installation of Unsloth Multi-GPU Support Package..."

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "Current Python version: $python_version"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✅ Python version meets requirements (>=3.8)"
else
    echo "❌ Python version too low, requires Python 3.8 or higher"
    exit 1
fi

# Check CUDA environment
echo "🔍 Checking CUDA environment..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️ No NVIDIA GPU detected, will install CPU version"
fi

# Install PyTorch
echo "📦 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "Installing GPU version of PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing CPU version of PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install Unsloth
echo "📦 Installing Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install other dependencies
echo "📦 Installing other dependencies..."
pip install transformers datasets accelerate
pip install psutil PyYAML tqdm numpy

# Optional dependencies
echo "❓ Install optional dependencies (TensorBoard, W&B)? [y/N]"
read -r install_optional

if [[ $install_optional =~ ^[Yy]$ ]]; then
    echo "📦 Installing optional dependencies..."
    pip install tensorboard wandb matplotlib seaborn
    echo "✅ Optional dependencies installed"
fi

# Verify installation
echo "🧪 Verifying installation..."
cd "$(dirname "$0")"

echo "Checking Python package imports..."
python3 -c "
try:
    import torch
    import transformers
    import unsloth
    import psutil
    import yaml
    print('✅ All required packages imported successfully')
except ImportError as e:
    print(f'❌ Package import failed: {e}')
    exit(1)
"

# Run quick tests
echo "🏃 Running quick tests..."
if python3 tests/run_all_tests.py --quick; then
    echo "✅ Quick tests passed"
else
    echo "⚠️ Quick tests failed, but basic installation may still work"
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📖 Quick start:"
echo "  import unsloth_multigpu as unsloth_multigpu"
echo "  unsloth_multigpu.enable_multi_gpu(num_gpus=4)"
echo ""
echo "📚 View examples:"
echo "  python examples/quick_start.py"
echo "  python examples/advanced_config.py"
echo ""
echo "🧪 Run full test suite:"
echo "  python tests/run_all_tests.py"
echo ""
echo "📞 If you encounter issues, please check README.md or run the test suite for troubleshooting"