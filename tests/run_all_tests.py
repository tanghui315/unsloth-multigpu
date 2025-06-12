#!/usr/bin/env python3
"""
Main program to run all Unsloth Multi-GPU tests
"""

import os
import sys
import time

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def run_all_tests():
    """Run all test suites"""
    print("🚀 Starting Unsloth Multi-GPU full test suite...")
    print("=" * 60)
    
    start_time = time.time()
    all_passed = True
    test_results = {}
    
    # 1. Core component tests
    print("\n📦 1. Core Component Tests")
    print("-" * 30)
    try:
        from test_core_components import run_core_tests
        result = run_core_tests()
        test_results['core_components'] = result
        all_passed &= result
    except Exception as e:
        print(f"❌ Core component tests failed: {e}")
        test_results['core_components'] = False
        all_passed = False
    
    # 2. MultiGPU Trainer tests
    print("\n🏋️ 2. MultiGPU Trainer Tests")
    print("-" * 30)
    try:
        from test_multi_gpu_trainer import run_trainer_tests
        result = run_trainer_tests()
        test_results['multi_gpu_trainer'] = result
        all_passed &= result
    except Exception as e:
        print(f"❌ MultiGPU Trainer tests failed: {e}")
        test_results['multi_gpu_trainer'] = False
        all_passed = False
    
    # 3. Hook system tests
    print("\n🔗 3. Hook System Tests")
    print("-" * 30)
    try:
        from test_hook_system import run_hook_system_tests
        result = run_hook_system_tests()
        test_results['hook_system'] = result
        all_passed &= result
    except Exception as e:
        print(f"❌ Hook system tests failed: {e}")
        test_results['hook_system'] = False
        all_passed = False
    
    # 4. Memory manager tests
    print("\n🧠 4. Memory Manager Tests")
    print("-" * 30)
    try:
        from test_memory_manager import run_memory_manager_tests
        result = run_memory_manager_tests()
        test_results['memory_manager'] = result
        all_passed &= result
    except Exception as e:
        print(f"❌ Memory manager tests failed: {e}")
        test_results['memory_manager'] = False
        all_passed = False
    
    # 5. Utils module tests
    print("\n🛠️ 5. Utils Module Tests")
    print("-" * 30)
    try:
        from test_utils import run_utils_tests
        result = run_utils_tests()
        test_results['utils'] = result
        all_passed &= result
    except Exception as e:
        print(f"❌ Utils module tests failed: {e}")
        test_results['utils'] = False
        all_passed = False
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    test_names = {
        'core_components': 'Core Components',
        'multi_gpu_trainer': 'MultiGPU Trainer',
        'hook_system': 'Hook System',
        'memory_manager': 'Memory Manager',
        'utils': 'Utils Module'
    }
    
    passed_count = 0
    total_count = len(test_results)
    
    for test_key, test_name in test_names.items():
        if test_key in test_results:
            status = "✅ Passed" if test_results[test_key] else "❌ Failed"
            print(f"{test_name:20} {status}")
            if test_results[test_key]:
                passed_count += 1
        else:
            print(f"{test_name:20} ⚠️ Skipped")
    
    print("-" * 60)
    print(f"Overall: {passed_count}/{total_count} test suites passed")
    print(f"Total time: {total_time:.2f} seconds")
    
    if all_passed:
        print("\n🎉 All tests passed! Unsloth Multi-GPU system is working correctly!")
        return True
    else:
        print(f"\n⚠️ {total_count - passed_count} test suites failed, please check the logs")
        return False


def run_quick_smoke_test():
    """Run a quick smoke test"""
    print("💨 Running quick smoke test...")
    
    try:
        # Test core imports
        from core import (BatchSharding, GradientAggregator, MemoryManager,
                          MultiGPUManager, MultiGPUTrainer)
        from hooks import LoaderHooks, TrainerHooks, TrainingHooks
        from utils import (ConfigManager, DeviceManager, MultiGPUConfig,
                           MultiGPULogger)
        
        print("✅ All modules imported successfully")
        
        # Test basic initialization
        config = MultiGPUConfig()
        memory_manager = MemoryManager(enable_auto_cleanup=False)
        device_manager = DeviceManager()
        logger = MultiGPULogger(enable_console=False, enable_file=False)
        
        print("✅ Basic components initialized successfully")
        
        # Test some basic functions
        cpu_memory_info = memory_manager.get_cpu_memory_info()
        device_summary = device_manager.get_device_summary()
        
        if 'error' not in cpu_memory_info:
            print("✅ CPU memory check passed")
        else:
            print("⚠️ CPU memory check has issues")
        
        print("✅ Smoke test passed")
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unsloth Multi-GPU Test Runner')
    parser.add_argument('--quick', action='store_true', help='Run quick smoke test')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_smoke_test()
    else:
        success = run_all_tests()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main() 