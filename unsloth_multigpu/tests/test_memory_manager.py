#!/usr/bin/env python3
"""
Memory management module unit tests
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add path to import core module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from core.memory_manager import MemoryManager
    MEMORY_MANAGER_AVAILABLE = True
except ImportError as e:
    MEMORY_MANAGER_AVAILABLE = False
    print(f"Warning: Memory manager not available: {e}")


class TestMemoryManager(unittest.TestCase):
    """Test MemoryManager class"""
    
    def setUp(self):
        if not MEMORY_MANAGER_AVAILABLE:
            self.skipTest("Memory manager not available")
        
        self.memory_manager = MemoryManager(
            warning_threshold=0.8,
            critical_threshold=0.9,
            enable_auto_cleanup=False  # Disable auto-cleanup during tests
        )
    
    def test_initialization(self):
        """Test memory manager initialization"""
        self.assertEqual(self.memory_manager.warning_threshold, 0.8)
        self.assertEqual(self.memory_manager.critical_threshold, 0.9)
        self.assertFalse(self.memory_manager.enable_auto_cleanup)
        
        # Check memory statistics initialization
        self.assertIn('peak_gpu_memory', self.memory_manager.memory_stats)
        self.assertIn('peak_cpu_memory', self.memory_manager.memory_stats)
        self.assertEqual(self.memory_manager.memory_stats['oom_count'], 0)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_gpu_memory_info_no_cuda(self, mock_cuda):
        """Test GPU memory information retrieval in no CUDA environment"""
        result = self.memory_manager.get_gpu_memory_info(0)
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'CUDA不可用')
    
    def test_cpu_memory_info(self):
        """Test CPU memory information retrieval"""
        result = self.memory_manager.get_cpu_memory_info()
        
        if 'error' not in result:
            # Check necessary fields
            expected_fields = [
                'total_memory_mb', 'available_memory_mb', 'used_memory_mb',
                'utilization', 'process_rss_mb', 'process_vms_mb', 'process_percent'
            ]
            
            for field in expected_fields:
                self.assertIn(field, result)
                self.assertIsInstance(result[field], (int, float))
            
            # Check reasonableness
            self.assertGreater(result['total_memory_mb'], 0)
            self.assertGreaterEqual(result['available_memory_mb'], 0)
            self.assertGreaterEqual(result['utilization'], 0)
            self.assertLessEqual(result['utilization'], 1)
    
    def test_memory_safety_check(self):
        """Test memory safety check"""
        result = self.memory_manager.check_memory_safety()
        
        # Check return structure
        expected_keys = ['safe', 'warnings', 'critical_issues', 'gpu_status', 'cpu_status', 'recommendations']
        for key in expected_keys:
            self.assertIn(key, result)
        
        self.assertIsInstance(result['safe'], bool)
        self.assertIsInstance(result['warnings'], list)
        self.assertIsInstance(result['critical_issues'], list)
        self.assertIsInstance(result['recommendations'], list)
    
    def test_cleanup_memory(self):
        """Test memory cleanup functionality"""
        result = self.memory_manager.cleanup_memory()
        
        # Check return structure
        expected_keys = ['success', 'cleaned_memory_mb', 'gpu_cleanup', 'cpu_cleanup', 'errors']
        for key in expected_keys:
            self.assertIn(key, result)
        
        self.assertIsInstance(result['success'], bool)
        self.assertIsInstance(result['cleaned_memory_mb'], (int, float))
        self.assertIsInstance(result['errors'], list)
        
        # Check statistics update
        self.assertGreater(self.memory_manager.memory_stats['cleanup_count'], 0)
    
    def test_oom_prevention(self):
        """Test OOM prevention functionality"""
        # Test small memory demand (should succeed)
        result = self.memory_manager.prevent_oom(1.0)  # 1MB
        self.assertIsInstance(result, bool)
        
        # Test extreme memory demand (should fail)
        result = self.memory_manager.prevent_oom(1000000.0)  # 1TB
        self.assertFalse(result)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_memory_guard_context_manager(self, mock_cuda):
        """Test memory guard context manager"""
        try:
            with self.memory_manager.memory_guard():
                # Simulate some memory operations
                dummy_data = [1] * 1000
                del dummy_data
        except Exception as e:
            self.fail(f"Memory guard context manager failed: {e}")
    
    def test_optimize_memory_usage(self):
        """Test memory usage optimization"""
        # Create mock model
        mock_model = MagicMock()
        mock_model.gradient_checkpointing_enable = MagicMock()
        
        result = self.memory_manager.optimize_memory_usage(
            model=mock_model,
            enable_gradient_checkpointing=True
        )
        
        # Check return structure
        expected_keys = ['optimizations_applied', 'memory_saved_mb', 'recommendations']
        for key in expected_keys:
            self.assertIn(key, result)
        
        self.assertIsInstance(result['optimizations_applied'], list)
        self.assertIsInstance(result['recommendations'], list)
    
    def test_memory_report(self):
        """Test memory report generation"""
        report = self.memory_manager.get_memory_report()
        
        # Check report structure
        expected_keys = ['timestamp', 'cpu_memory', 'gpu_memory', 'memory_stats', 'system_info']
        for key in expected_keys:
            self.assertIn(key, report)
        
        self.assertIsInstance(report['timestamp'], (int, float))
        self.assertIsInstance(report['memory_stats'], dict)
        self.assertIsInstance(report['system_info'], dict)


class TestMemoryManagerIntegration(unittest.TestCase):
    """Test memory manager integration functionality"""
    
    def setUp(self):
        if not MEMORY_MANAGER_AVAILABLE:
            self.skipTest("Memory manager not available")
    
    def test_memory_manager_lifecycle(self):
        """Test memory manager lifecycle"""
        # Create manager
        manager = MemoryManager(enable_auto_cleanup=False)
        
        # Perform some operations
        manager.get_cpu_memory_info()
        manager.cleanup_memory()
        safety_check = manager.check_memory_safety()
        
        # Check statistics
        stats = manager.memory_stats
        self.assertGreaterEqual(stats['cleanup_count'], 1)
        
        # Generate report
        report = manager.get_memory_report()
        self.assertIsInstance(report, dict)
    
    def test_memory_thresholds(self):
        """Test memory threshold functionality"""
        # Test different threshold settings
        manager_strict = MemoryManager(
            warning_threshold=0.5,
            critical_threshold=0.7
        )
        
        manager_relaxed = MemoryManager(
            warning_threshold=0.9,
            critical_threshold=0.95
        )
        
        # Two managers should have different thresholds
        self.assertNotEqual(
            manager_strict.warning_threshold,
            manager_relaxed.warning_threshold
        )
    
    def test_error_handling(self):
        """Test error handling"""
        manager = MemoryManager()
        
        # Test invalid GPU memory query with device ID
        result = manager.get_gpu_memory_info(999)
        self.assertIn('error', result)


def run_memory_manager_tests():
    """Run memory manager tests"""
    if not MEMORY_MANAGER_AVAILABLE:
        print("❌ Memory manager not available, skipping tests")
        return False
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMemoryManager,
        TestMemoryManagerIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print results
    if result.wasSuccessful():
        print(f"\n✅ All memory manager tests passed! ({result.testsRun} tests)")
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
    run_memory_manager_tests() 