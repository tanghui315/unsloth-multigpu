#!/usr/bin/env python3
"""
Unsloth Multi-GPU Installation Verification Script
Check if all dependencies are correctly installed and available
"""

import os
import sys
import traceback
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
parent_dir = os.path.dirname(project_root)
sys.path.insert(0, parent_dir)

def check_import(module_name: str, import_from: str = None) -> Tuple[bool, str]:
    """
    Check module import
    
    Args:
        module_name: Module name
        import_from: Import from which package (optional)
        
    Returns:
        Tuple[bool, str]: (Whether successful, error message or version)
    """
    try:
        if import_from:
            exec(f"from {import_from} import {module_name}")
        else:
            module = __import__(module_name)
            
        # 尝试获取版本信息
        try:
            if import_from:
                version = eval(f"{import_from}.__version__")
            else:
                version = getattr(module, '__version__', 'Unknown')
            return True, f"v{version}"
        except:
            return True, "OK"
            
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"

def check_gpu_environment() -> Dict[str, any]:
    """Check GPU environment"""
    gpu_info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_names': [],
        'current_device': None,
        'error': None
    }
    
    try:
        import torch
        gpu_info['cuda_available'] = torch.cuda.is_available()
        
        if gpu_info['cuda_available']:
            gpu_info['gpu_count'] = torch.cuda.device_count()
            gpu_info['current_device'] = torch.cuda.current_device()
            gpu_info['gpu_names'] = [
                torch.cuda.get_device_name(i) 
                for i in range(gpu_info['gpu_count'])
            ]
    except Exception as e:
        gpu_info['error'] = str(e)
    
    return gpu_info

def check_unsloth_with_fallback() -> List[Tuple[str, bool, str]]:
    """Check Unsloth, handle environments without GPU"""
    results = []
    
    # First try to import unsloth package itself
    try:
        import unsloth
        results.append(("unsloth", True, f"v{getattr(unsloth, '__version__', 'Unknown')}"))
    except ImportError as e:
        results.append(("unsloth", False, f"Import failed: {e}"))
        return results
    
    # Try to import specific components (may fail on GPUs)
    unsloth_components = [
        "FastLanguageModel",
        "unsloth_train",
    ]
    
    for component in unsloth_components:
        try:
            # Use getattr instead of dynamic import to avoid GPU check
            if hasattr(unsloth, component):
                results.append((component, True, "Available"))
            else:
                # Try to import from submodule
                exec(f"from unsloth import {component}")
                results.append((component, True, "Available"))
        except Exception as e:
            error_msg = str(e)
            if "NVIDIA GPUs" in error_msg or "Intel GPUs" in error_msg:
                results.append((component, True, "Available (requires GPU to run)"))
            else:
                results.append((component, False, f"Import failed: {error_msg}"))
    
    return results

def main():
    """Main verification function"""
    print("🔍 Unsloth Multi-GPU Installation Verification")
    print("=" * 50)
    
    # Check Python version
    print(f"\n📋 Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python version too low, need 3.8 or higher")
        return False
    else:
        print("✅ Python version meets requirements")
    
    # Check core dependencies
    print("\n📦 Checking core dependencies:")
    core_deps = [
        ("torch", None),
        ("transformers", None),
        ("datasets", None),
        ("accelerate", None),
        ("psutil", None),
        ("yaml", None),
        ("numpy", None),
        ("tqdm", None),
    ]
    
    failed_deps = []
    for dep, from_module in core_deps:
        success, result = check_import(dep, from_module)
        status = "✅" if success else "❌"
        print(f"  {status} {dep}: {result}")
        if not success:
            failed_deps.append(dep)
    
    # Check Unsloth
    print("\n🦄 Checking Unsloth:")
    results = check_unsloth_with_fallback()
    
    for component, success, result in results:
        status = "✅" if success else "❌"
        print(f"  {status} {component}: {result}")
        if not success:
            failed_deps.append(f"unsloth.{component}")
    
    # Check project modules
    print("\n🔧 Checking project modules:")
    project_modules = [
        ("unsloth_multigpu_prototype", None),
        ("MultiGPUManager", "unsloth_multigpu_prototype.core"),
        ("ConfigManager", "unsloth_multigpu_prototype.utils"),
        ("MultiGPULogger", "unsloth_multigpu_prototype.utils"),
    ]
    
    for module, from_package in project_modules:
        success, result = check_import(module, from_package)
        status = "✅" if success else "❌"
        print(f"  {status} {module}: {result}")
        if not success:
            failed_deps.append(f"{from_package}.{module}" if from_package else module)
    
    # Check GPU environment
    print("\n🎮 Checking GPU environment:")
    gpu_info = check_gpu_environment()
    
    if gpu_info['error']:
        print(f"❌ GPU check failed: {gpu_info['error']}")
    else:
        print(f"  CUDA available: {'✅' if gpu_info['cuda_available'] else '❌'}")
        print(f"  GPU count: {gpu_info['gpu_count']}")
        
        if gpu_info['gpu_names']:
            print("  GPU list:")
            for i, name in enumerate(gpu_info['gpu_names']):
                print(f"    GPU {i}: {name}")
        
        if gpu_info['gpu_count'] >= 2:
            print("✅ Supports multi-GPU training")
        elif gpu_info['gpu_count'] == 1:
            print("⚠️ Only 1 GPU, multi-GPU functionality will be unavailable")
        else:
            print("❌ No available GPUs")
    
    # Test main functionality
    print("\n🧪 Testing main functionality:")
    
    try:
        import unsloth_multigpu_prototype as unsloth_multigpu
        print("✅ Main module import successful")
        
        # Test status query
        status = unsloth_multigpu.get_multi_gpu_status()
        print(f"✅ Status query successful: {status.get('enabled', 'Unknown')}")
        
        # Test enable function (not actually enabling)
        if gpu_info['gpu_count'] >= 2:
            print("✅ Multi-GPU functionality available")
        else:
            print("⚠️ Multi-GPU functionality unavailable (insufficient GPU count)")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        failed_deps.append("unsloth_multigpu_prototype")
    
    # Summarize results
    print("\n" + "=" * 50)
    if failed_deps:
        print(f"❌ Verification failed, missing dependencies: {', '.join(failed_deps)}")
        print("\n🔧 Repair suggestions:")
        print("1. Run installation script: ./install.sh")
        print("2. Manually install missing packages:")
        for dep in failed_deps:
            if 'unsloth' in dep.lower():
                print(f"   pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"")
            else:
                print(f"   pip install {dep}")
        return False
    else:
        print("✅ All verification passed! Unsloth Multi-GPU support package installed correctly")
        print("\n🚀 Quick start:")
        print("   import unsloth_multigpu_prototype as unsloth_multigpu")
        print("   unsloth_multigpu.enable_multi_gpu()")
        print("\n📚 View example:")
        print("   python examples/quick_start.py")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 