#!/usr/bin/env python3
"""
Utility module unit tests
Including tests for ConfigManager, DeviceManager, and MultiGPULogger
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# 添加路径以导入utils模块
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from utils import (ConfigManager, DeviceManager, MultiGPUConfig,
                       MultiGPULogger, get_default_config)
    UTILS_AVAILABLE = True
except ImportError as e:
    UTILS_AVAILABLE = False
    print(f"Warning: Utils not available: {e}")


class TestMultiGPUConfig(unittest.TestCase):
    """Test MultiGPUConfig class"""
    
    def setUp(self):
        if not UTILS_AVAILABLE:
            self.skipTest("Utils not available")
    
    def test_default_config(self):
        """Test default configuration"""
        config = MultiGPUConfig()
        
        # Check default values
        # Note: In no GPU environment, enabled will automatically be set to False
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            self.assertTrue(config.enabled)
        else:
            # Should automatically disable in no GPU or insufficient GPU environment
            self.assertFalse(config.enabled)
        
        self.assertEqual(config.batch_size_per_gpu, 2)
        self.assertEqual(config.gradient_aggregation, "mean")
        self.assertTrue(config.enable_gradient_checkpointing)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid configuration
        valid_config = MultiGPUConfig(
            gradient_aggregation="mean",
            batch_sharding_strategy="uniform"
        )
        # Should not raise an exception
        
        # Invalid configuration
        with self.assertRaises(ValueError):
            MultiGPUConfig(gradient_aggregation="invalid")
        
        with self.assertRaises(ValueError):
            MultiGPUConfig(batch_sharding_strategy="invalid")
    
    def test_config_auto_detection(self):
        """Test automatic detection functionality"""
        config = MultiGPUConfig()
        
        # Check GPU number automatic detection
        self.assertIsInstance(config.num_gpus, int)
        self.assertGreaterEqual(config.num_gpus, 0)
        
        # Check device ID automatic selection
        if config.num_gpus > 0:
            self.assertIsInstance(config.device_ids, list)
            self.assertLessEqual(len(config.device_ids), config.num_gpus)


class TestConfigManager(unittest.TestCase):
    """Test ConfigManager class"""
    
    def setUp(self):
        if not UTILS_AVAILABLE:
            self.skipTest("Utils not available")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(config_dir=self.temp_dir)
    
    def tearDown(self):
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_config(self):
        """Test configuration saving and loading"""
        # Create test configuration
        original_config = MultiGPUConfig(
            batch_size_per_gpu=4,
            learning_rate=1e-4,
            gradient_aggregation="weighted_mean"
        )
        
        # Save configuration
        saved_path = self.config_manager.save_config(original_config, "test_config")
        self.assertTrue(os.path.exists(saved_path))
        
        # Load configuration
        loaded_config = self.config_manager.load_config("test_config")
        
        # Verify configuration consistency
        self.assertEqual(loaded_config.batch_size_per_gpu, 4)
        self.assertEqual(loaded_config.learning_rate, 1e-4)
        self.assertEqual(loaded_config.gradient_aggregation, "weighted_mean")
    
    def test_create_default_configs(self):
        """Test creating default configurations"""
        self.config_manager.create_default_configs()
        
        # Check generated configuration files
        configs = self.config_manager.list_configs()
        expected_configs = ["default_config", "high_performance_config", 
                          "memory_efficient_config", "debug_config"]
        
        for expected_config in expected_configs:
            self.assertIn(expected_config, configs)
    
    def test_validate_config(self):
        """Test configuration validation"""
        # Normal configuration
        normal_config = MultiGPUConfig()
        validation_result = self.config_manager.validate_config(normal_config)
        
        self.assertIn('valid', validation_result)
        self.assertIn('warnings', validation_result)
        self.assertIn('errors', validation_result)
        self.assertIn('suggestions', validation_result)
    
    def test_compare_configs(self):
        """Test configuration comparison"""
        config1 = MultiGPUConfig(batch_size_per_gpu=2)
        config2 = MultiGPUConfig(batch_size_per_gpu=4)
        
        comparison = self.config_manager.compare_configs(config1, config2)
        
        self.assertIn('identical', comparison)
        self.assertIn('differences', comparison)
        self.assertFalse(comparison['identical'])
        self.assertIn('batch_size_per_gpu', comparison['differences'])
    
    def test_get_optimal_config(self):
        """Test getting optimal configuration"""
        # Test different types of optimal configurations
        fast_config = self.config_manager.get_optimal_config(training_type="fast")
        memory_config = self.config_manager.get_optimal_config(training_type="memory_efficient")
        standard_config = self.config_manager.get_optimal_config(training_type="standard")
        
        # Verify configuration differences
        self.assertIsInstance(fast_config, MultiGPUConfig)
        self.assertIsInstance(memory_config, MultiGPUConfig)
        self.assertIsInstance(standard_config, MultiGPUConfig)


class TestDeviceManager(unittest.TestCase):
    """Test DeviceManager class"""
    
    def setUp(self):
        if not UTILS_AVAILABLE:
            self.skipTest("Utils not available")
        
        self.device_manager = DeviceManager()
    
    def test_device_manager_initialization(self):
        """Test device manager initialization"""
        self.assertIsInstance(self.device_manager.device_info, dict)
        self.assertFalse(self.device_manager.monitoring_active)
    
    def test_get_available_devices(self):
        """Test getting available devices"""
        devices = self.device_manager.get_available_devices()
        self.assertIsInstance(devices, list)
        
        # All device IDs should be non-negative integers
        for device_id in devices:
            self.assertIsInstance(device_id, int)
            self.assertGreaterEqual(device_id, 0)
    
    def test_get_optimal_device_assignment(self):
        """Test getting optimal device assignment"""
        # Request 2 GPUs
        assignment = self.device_manager.get_optimal_device_assignment(num_gpus=2)
        self.assertIsInstance(assignment, list)
        
        # 如果有可用GPU，分配数量应该不超过请求数量
        available_devices = self.device_manager.get_available_devices()
        if available_devices:
            self.assertLessEqual(len(assignment), min(2, len(available_devices)))
    
    def test_check_device_compatibility(self):
        """Test device compatibility check"""
        available_devices = self.device_manager.get_available_devices()
        
        if len(available_devices) >= 2:
            # Test compatibility of multiple devices
            compatibility = self.device_manager.check_device_compatibility(available_devices[:2])
            
            expected_keys = ['compatible', 'warnings', 'errors', 'device_comparison']
            for key in expected_keys:
                self.assertIn(key, compatibility)
            
            self.assertIsInstance(compatibility['compatible'], bool)
            self.assertIsInstance(compatibility['warnings'], list)
            self.assertIsInstance(compatibility['errors'], list)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_device_manager_no_cuda(self, mock_cuda):
        """Test device manager in no CUDA environment"""
        manager = DeviceManager()
        devices = manager.get_available_devices()
        self.assertEqual(len(devices), 0)
    
    def test_device_summary(self):
        """Test device summary"""
        summary = self.device_manager.get_device_summary()
        
        expected_keys = ['cuda_available', 'device_count', 'current_device', 
                        'devices', 'monitoring_active']
        for key in expected_keys:
            self.assertIn(key, summary)


class TestMultiGPULogger(unittest.TestCase):
    """Test MultiGPULogger class"""
    
    def setUp(self):
        if not UTILS_AVAILABLE:
            self.skipTest("Utils not available")
        
        # Create temporary log directory
        self.temp_log_dir = tempfile.mkdtemp()
        self.logger = MultiGPULogger(
            log_dir=self.temp_log_dir,
            enable_file=True,
            enable_console=False  # Disable console output during tests
        )
    
    def tearDown(self):
        # Clean up temporary directory
        if os.path.exists(self.temp_log_dir):
            shutil.rmtree(self.temp_log_dir)
    
    def test_logger_initialization(self):
        """Test logger initialization"""
        self.assertEqual(self.logger.log_dir, self.temp_log_dir)
        self.assertIn('start_time', self.logger.performance_metrics)
        self.assertIn('step_times', self.logger.performance_metrics)
    
    def test_log_training_step(self):
        """Test training step logging"""
        # Log several training steps
        for i in range(3):
            self.logger.log_training_step(
                step=i,
                loss=0.1 * (3-i),
                metrics={'accuracy': 0.8 + 0.05*i}
            )
        
        # Check performance metrics update
        self.assertEqual(len(self.logger.performance_metrics['step_times']), 3)
        
        # Check throughput history
        if len(self.logger.performance_metrics['throughput_history']) > 0:
            throughput = self.logger.performance_metrics['throughput_history']
            self.assertTrue(all(t > 0 for t in throughput))
    
    def test_log_gpu_status(self):
        """Test GPU status logging"""
        mock_stats = {
            0: {'memory_allocated_mb': 1024, 'memory_reserved_mb': 2048},
            1: {'memory_allocated_mb': 512, 'memory_reserved_mb': 1024}
        }
        
        self.logger.log_gpu_status(mock_stats)
        # Should not raise an exception
    
    def test_log_error_and_warning(self):
        """Test error and warning logging"""
        # Log warning
        self.logger.log_warning("Test warning", context={'test': True})
        self.assertEqual(self.logger.performance_metrics['warning_count'], 1)
        
        # Log error
        test_error = ValueError("Test error")
        self.logger.log_error(test_error, context={'error_test': True})
        self.assertEqual(self.logger.performance_metrics['error_count'], 1)
    
    def test_performance_stats(self):
        """Test performance statistics"""
        # Log some activity
        self.logger.log_training_step(1, 0.5)
        self.logger.log_warning("Test warning")
        
        stats = self.logger.get_performance_stats()
        
        expected_keys = ['start_time', 'total_duration', 'warning_count', 'error_count']
        for key in expected_keys:
            self.assertIn(key, stats)
        
        self.assertGreater(stats['total_duration'], 0)
    
    def test_log_duration_context_manager(self):
        """Test duration logging context manager"""
        with self.logger.log_duration("test_operation"):
            # Simulate some operations
            import time
            time.sleep(0.01)
        
        # Should not raise an exception
    
    def test_log_memory_usage(self):
        """Test memory usage logging"""
        # This method should be safe to return in no CUDA environment
        self.logger.log_memory_usage()
        # Should not raise an exception


class TestUtilsIntegration(unittest.TestCase):
    """Test utility module integration"""
    
    def setUp(self):
        if not UTILS_AVAILABLE:
            self.skipTest("Utils not available")
    
    def test_config_device_logger_integration(self):
        """Test configuration, device, and logger system integration"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 1. Create configuration
            config = get_default_config()
            
            # 2. Save and load configuration
            config_manager = ConfigManager(config_dir=temp_dir)
            config_manager.save_config(config, "integration_test")
            loaded_config = config_manager.load_config("integration_test")
            
            # 3. Device management
            device_manager = DeviceManager()
            available_devices = device_manager.get_available_devices()
            
            # 4. Logger system
            logger = MultiGPULogger(
                log_dir=os.path.join(temp_dir, "logs"),
                enable_console=False
            )
            
            # 5. Integration test
            logger.log_training_step(1, 0.5)
            
            # Verify all components are working
            self.assertIsInstance(loaded_config, MultiGPUConfig)
            self.assertIsInstance(available_devices, list)
            self.assertGreater(len(logger.performance_metrics['step_times']), 0)
            
        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def test_utility_functions(self):
        """Test utility functions"""
        # Test default configuration retrieval
        default_config = get_default_config()
        self.assertIsInstance(default_config, MultiGPUConfig)
        
        # Test environment configuration
        from utils.config_utils import setup_config_for_environment
        env_config = setup_config_for_environment()
        self.assertIsInstance(env_config, MultiGPUConfig)


def run_utils_tests():
    """Run utility module tests"""
    if not UTILS_AVAILABLE:
        print("❌ Utils not available, skipping tests")
        return False
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMultiGPUConfig,
        TestConfigManager,
        TestDeviceManager,
        TestMultiGPULogger,
        TestUtilsIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print results
    if result.wasSuccessful():
        print(f"\n✅ All utils tests passed! ({result.testsRun} tests)")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")
        
        # Print failure details 
        for test, traceback in result.failures:
            print(f"\nFAILURE: {test}")
            print(traceback)
        
        for test, traceback in result.errors:
            print(f"\nERROR: {test}")
            print(traceback)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    run_utils_tests() 