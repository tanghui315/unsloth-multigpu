#!/usr/bin/env python3
"""
Hook system unit tests
Test the functionality of the modular Hook system
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add path to import hooks module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from hooks import LoaderHooks, TrainerHooks, TrainingHooks
    HOOKS_AVAILABLE = True
except ImportError as e:
    HOOKS_AVAILABLE = False
    print(f"Warning: Hook components not available: {e}")


class TestTrainingHooks(unittest.TestCase):
    """Test TrainingHooks class"""
    
    def setUp(self):
        if not HOOKS_AVAILABLE:
            self.skipTest("Hook components not available")
        
        self.training_hooks = TrainingHooks()
    
    def test_initialization(self):
        """Test TrainingHooks initialization"""
        self.assertFalse(self.training_hooks.hooks_applied)
        self.assertEqual(len(self.training_hooks.original_functions), 0)
    
    def test_hook_status(self):
        """Test Hook status retrieval"""
        status = self.training_hooks.get_hook_status()
        
        self.assertIn('hooks_applied', status)
        self.assertIn('active_hooks', status)
        self.assertIn('num_hooks', status)
        self.assertIn('training_hooks', status)
        
        self.assertFalse(status['hooks_applied'])
        self.assertEqual(status['num_hooks'], 0)
        self.assertEqual(status['training_hooks'], ['get_max_steps', 'unsloth_train'])
    
    @patch.dict(os.environ, {'UNSLOTH_MULTIGPU_ENABLED': '1', 'UNSLOTH_MULTIGPU_NUM_GPUS': '2'})
    def test_multi_gpu_training_execution_structure(self):
        """Test multi-GPU training execution structure (without actual execution)"""
        # Create a mock trainer
        mock_trainer = MagicMock()
        mock_trainer.model = MagicMock()
        mock_trainer.train_dataloader = MagicMock()
        mock_trainer.args = MagicMock()
        mock_trainer.args.max_steps = 100
        mock_trainer.args.num_train_epochs = 1
        
        # Test optimizer configuration extraction
        config = self.training_hooks._extract_optimizer_config(mock_trainer)
        
        self.assertIn('class', config)
        self.assertIn('kwargs', config)
        self.assertIn('lr', config['kwargs'])
        self.assertIn('weight_decay', config['kwargs'])
    
    def test_remove_hooks_when_not_applied(self):
        """Test removing Hook when it is not applied"""
        result = self.training_hooks.remove_hooks()
        self.assertTrue(result)  # Should return True on success


class TestLoaderHooks(unittest.TestCase):
    """Test LoaderHooks class"""
    
    def setUp(self):
        if not HOOKS_AVAILABLE:
            self.skipTest("Hook components not available")
        
        self.loader_hooks = LoaderHooks()
    
    def test_initialization(self):
        """Test LoaderHooks initialization"""
        self.assertFalse(self.loader_hooks.hooks_applied)
        self.assertEqual(len(self.loader_hooks.original_functions), 0)
    
    def test_hook_status(self):
        """Test Hook status retrieval"""
        status = self.loader_hooks.get_hook_status()
        
        self.assertIn('hooks_applied', status)
        self.assertIn('active_hooks', status)
        self.assertIn('num_hooks', status)
        self.assertIn('loader_hooks', status)
        
        self.assertFalse(status['hooks_applied'])
        self.assertEqual(status['num_hooks'], 0)
        self.assertEqual(status['loader_hooks'], ['from_pretrained'])
    
    def test_multi_gpu_config_validation(self):
        """Test multi-GPU configuration validation"""
        # Valid configuration
        valid_config = {
            'gradient_checkpointing': True,
            'mixed_precision': True,
            'gradient_aggregation': 'mean'
        }
        
        result = self.loader_hooks._validate_multi_gpu_config(valid_config)
        self.assertTrue(result)
        
        # Invalid configuration
        invalid_config = {
            'gradient_aggregation': 'invalid_method'
        }
        
        result = self.loader_hooks._validate_multi_gpu_config(invalid_config)
        self.assertFalse(result)
    
    def test_model_multi_gpu_info_none(self):
        """Test model multi-GPU information retrieval for models that do not support multi-GPU"""
        mock_model = MagicMock()
        # Model without multi-GPU attributes
        del mock_model._unsloth_multi_gpu_enabled
        
        result = self.loader_hooks.get_model_multi_gpu_info(mock_model)
        self.assertIsNone(result)
    
    def test_model_multi_gpu_info_available(self):
        """Test model multi-GPU information retrieval for models that support multi-GPU"""
        mock_model = MagicMock()
        mock_model._unsloth_multi_gpu_enabled = True
        mock_model._unsloth_target_num_gpus = 4
        mock_model._unsloth_multi_gpu_config = {'test': 'config'}
        
        result = self.loader_hooks.get_model_multi_gpu_info(mock_model)
        
        self.assertIsNotNone(result)
        self.assertTrue(result['enabled'])
        self.assertEqual(result['num_gpus'], 4)
        self.assertEqual(result['config'], {'test': 'config'})


class TestTrainerHooks(unittest.TestCase):
    """Test TrainerHooks class"""
    
    def setUp(self):
        if not HOOKS_AVAILABLE:
            self.skipTest("Hook components not available")
        
        self.trainer_hooks = TrainerHooks()
    
    def test_initialization(self):
        """Test TrainerHooks initialization"""
        self.assertFalse(self.trainer_hooks.hooks_applied)
        self.assertEqual(len(self.trainer_hooks.original_functions), 0)
        self.assertIn('hook_overhead_ms', self.trainer_hooks.performance_stats)
    
    def test_hook_status(self):
        """Test Hook status retrieval"""
        status = self.trainer_hooks.get_hook_status()
        
        self.assertIn('hooks_applied', status)
        self.assertIn('active_hooks', status)
        self.assertIn('num_hooks', status)
        self.assertIn('trainer_hooks', status)
        self.assertIn('performance_stats', status)
        
        self.assertFalse(status['hooks_applied'])
        self.assertEqual(status['num_hooks'], 0)
        expected_hooks = ['trainer_init', 'training_step', 'evaluate', 'save_model']
        self.assertEqual(status['trainer_hooks'], expected_hooks)
    
    def test_performance_stats(self):
        """Test performance statistics functionality"""
        # Initial state
        stats = self.trainer_hooks.get_performance_stats()
        self.assertEqual(stats['total_hook_calls'], 0)
        self.assertEqual(stats['avg_hook_overhead_ms'], 0.0)
        self.assertEqual(stats['success_rate'], 0.0)
        
        # Simulate some performance data
        self.trainer_hooks._update_performance_stats(0.001, success=True)
        self.trainer_hooks._update_performance_stats(0.002, success=False)
        
        stats = self.trainer_hooks.get_performance_stats()
        self.assertEqual(stats['total_hook_calls'], 2)
        self.assertEqual(stats['successful_hooks'], 1)
        self.assertEqual(stats['failed_hooks'], 1)
        self.assertEqual(stats['success_rate'], 0.5)
        
        # Reset statistics
        self.trainer_hooks.reset_performance_stats()
        stats = self.trainer_hooks.get_performance_stats()
        self.assertEqual(stats['total_hook_calls'], 0)
    
    def test_prepare_inputs(self):
        """Test input data preparation"""
        # Dictionary input
        dict_input = {'input_ids': [1, 2, 3], 'labels': [4, 5, 6]}
        result = self.trainer_hooks._prepare_inputs(dict_input)
        self.assertEqual(result, dict_input)
        
        # Object input
        class MockInput:
            def __init__(self):
                self.input_ids = [1, 2, 3]
                self.labels = [4, 5, 6]
        
        obj_input = MockInput()
        result = self.trainer_hooks._prepare_inputs(obj_input)
        self.assertIn('input_ids', result)
        self.assertIn('labels', result)
        
        # Other type input
        other_input = [1, 2, 3]
        result = self.trainer_hooks._prepare_inputs(other_input)
        self.assertEqual(result, {'input_ids': other_input})


class TestHookSystemIntegration(unittest.TestCase):
    """Test Hook system integration functionality"""
    
    def setUp(self):
        if not HOOKS_AVAILABLE:
            self.skipTest("Hook components not available")
        
        self.training_hooks = TrainingHooks()
        self.loader_hooks = LoaderHooks()
        self.trainer_hooks = TrainerHooks()
    
    def test_multiple_hook_managers(self):
        """Test multiple Hook manager coordination"""
        # All managers should be independent
        self.assertNotEqual(id(self.training_hooks), id(self.loader_hooks))
        self.assertNotEqual(id(self.loader_hooks), id(self.trainer_hooks))
        
        # Check status independence
        training_status = self.training_hooks.get_hook_status()
        loader_status = self.loader_hooks.get_hook_status()
        trainer_status = self.trainer_hooks.get_hook_status()
        
        self.assertNotEqual(training_status['training_hooks'], loader_status['loader_hooks'])
        self.assertNotEqual(loader_status['loader_hooks'], trainer_status['trainer_hooks'])
    
    def test_hook_lifecycle(self):
        """Test Hook lifecycle management"""
        # Initial state
        self.assertFalse(self.training_hooks.hooks_applied)
        
        # Try to remove Hooks that are not applied (should be safe failure)
        result = self.training_hooks.remove_hooks()
        self.assertTrue(result)
        
        # State should remain unchanged
        self.assertFalse(self.training_hooks.hooks_applied)
    
    def test_error_handling(self):
        """Test error handling mechanism"""
        # Test behavior when there is no related module
        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            # This should not crash but gracefully handle the error
            result = self.training_hooks._hook_get_max_steps()
            # Since the module does not exist, it should return True (non-fatal error)
            self.assertTrue(result)


class TestHookManagersCoordination(unittest.TestCase):
    """Test Hook manager coordination functionality"""
    
    def setUp(self):
        if not HOOKS_AVAILABLE:
            self.skipTest("Hook components not available")
    
    def test_hook_manager_creation(self):
        """Test Hook manager creation"""
        # Create all types of Hook managers
        managers = {
            'training': TrainingHooks(),
            'loader': LoaderHooks(),
            'trainer': TrainerHooks()
        }
        
        # Verify all managers are correctly created
        for name, manager in managers.items():
            self.assertIsNotNone(manager)
            self.assertFalse(manager.hooks_applied)
            
            # Verify status retrieval method exists
            self.assertTrue(hasattr(manager, 'get_hook_status'))
            status = manager.get_hook_status()
            self.assertIsInstance(status, dict)
    
    @patch.dict(os.environ, {'UNSLOTH_MULTIGPU_ENABLED': '1'})
    def test_environment_variable_handling(self):
        """Test environment variable handling"""
        # Test impact of multi-GPU environment variables
        self.assertEqual(os.environ.get('UNSLOTH_MULTIGPU_ENABLED'), '1')
        
        # This should affect Hook behavior (although it will not actually apply in the test environment)
        training_hooks = TrainingHooks()
        self.assertIsNotNone(training_hooks)


def run_hook_system_tests():
    """Run Hook system tests"""
    if not HOOKS_AVAILABLE:
        print("❌ Hook components not available, skipping hook system tests")
        return False
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTrainingHooks,
        TestLoaderHooks,
        TestTrainerHooks,
        TestHookSystemIntegration,
        TestHookManagersCoordination
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print results
    if result.wasSuccessful():
        print(f"\n✅ All hook system tests passed! ({result.testsRun} tests)")
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
    run_hook_system_tests() 