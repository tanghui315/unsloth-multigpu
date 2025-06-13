#!/usr/bin/env python3
"""
Core component unit tests
Test the functionality of MultiGPUManager, BatchSharding, and GradientAggregator
"""

import os
import sys
import unittest

import torch
import torch.nn as nn

# Add path to import core module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from core import (AggregationMethod, BatchSharding, GradientAggregator,
                      MultiGPUManager)
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    print(f"Warning: Core components not available: {e}")

class TestBatchSharding(unittest.TestCase):
    """Test BatchSharding component"""
    
    def setUp(self):
        if not CORE_AVAILABLE:
            self.skipTest("Core components not available")
        self.num_gpus = 4
        self.sharder = BatchSharding(self.num_gpus)
    
    def test_normal_batch_splitting(self):
        """Test normal batch splitting"""
        batch_size = 16
        batch = {
            'input_ids': torch.randint(0, 1000, (batch_size, 128)),
            'labels': torch.randint(0, 1000, (batch_size, 128)),
            'attention_mask': torch.ones(batch_size, 128)
        }
        
        shards = self.sharder.split_batch(batch)
        
        # Verify shard count
        self.assertEqual(len(shards), self.num_gpus)
        
        # Verify total sample count remains consistent
        total_samples = sum(shard['input_ids'].size(0) for shard in shards)
        self.assertEqual(total_samples, batch_size)
        
        # Verify each shard has correct keys
        for shard in shards:
            self.assertIn('input_ids', shard)
            self.assertIn('labels', shard)
            self.assertIn('attention_mask', shard)
    
    def test_small_batch_splitting(self):
        """Test small batch splitting (batch size < number of GPUs)"""
        batch_size = 2  # Less than GPU count
        batch = {
            'input_ids': torch.randint(0, 1000, (batch_size, 128)),
            'labels': torch.randint(0, 1000, (batch_size, 128))
        }
        
        shards = self.sharder.split_batch(batch)
        
        # Verify shard count
        self.assertEqual(len(shards), self.num_gpus)
        
        # Verify first two shards have data, last two are empty
        self.assertEqual(shards[0]['input_ids'].size(0), 1)
        self.assertEqual(shards[1]['input_ids'].size(0), 1)
        self.assertEqual(shards[2]['input_ids'].size(0), 0)
        self.assertEqual(shards[3]['input_ids'].size(0), 0)
    
    def test_empty_batch_handling(self):
        """Test empty batch handling"""
        empty_batch = {}
        shards = self.sharder.split_batch(empty_batch)
        
        self.assertEqual(len(shards), self.num_gpus)
        for shard in shards:
            self.assertEqual(shard, {})
    
    def test_output_gathering(self):
        """Test output gathering"""
        # Test numerical list
        outputs = [1.0, 2.0, 3.0, 4.0]
        result = self.sharder.gather_outputs(outputs)
        self.assertAlmostEqual(result, 2.5)  # Average
        
        # Test tensor list
        tensors = [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6])]
        result = self.sharder.gather_outputs(tensors)
        expected = torch.tensor([1, 2, 3, 4, 5, 6])
        self.assertTrue(torch.equal(result, expected))

class TestGradientAggregator(unittest.TestCase):
    """Test GradientAggregator component"""
    
    def setUp(self):
        if not CORE_AVAILABLE:
            self.skipTest("Core components not available")
        self.aggregator = GradientAggregator(
            aggregation_method=AggregationMethod.MEAN,
            enable_consistency_check=True,
            enable_gradient_clipping=False  # Disable gradient clipping to avoid test interference
        )
    
    def test_mean_aggregation(self):
        """Test mean aggregation"""
        # Create simulated gradients
        gradients_per_gpu = [
            {'param1': torch.tensor([1.0, 2.0]), 'param2': torch.tensor([3.0, 4.0])},
            {'param1': torch.tensor([5.0, 6.0]), 'param2': torch.tensor([7.0, 8.0])},
        ]
        
        result = self.aggregator.aggregate(gradients_per_gpu)
        
        # Verify results
        expected_param1 = torch.tensor([3.0, 4.0])  # (1+5)/2, (2+6)/2
        expected_param2 = torch.tensor([5.0, 6.0])  # (3+7)/2, (4+8)/2
        
        self.assertTrue(torch.allclose(result['param1'], expected_param1, rtol=1e-3))
        self.assertTrue(torch.allclose(result['param2'], expected_param2, rtol=1e-3))
    
    def test_weighted_aggregation(self):
        """Test weighted aggregation"""
        aggregator = GradientAggregator(
            aggregation_method=AggregationMethod.WEIGHTED_MEAN,
            enable_gradient_clipping=False  # Disable gradient clipping
        )
        
        gradients_per_gpu = [
            {'param1': torch.tensor([2.0, 4.0])},
            {'param1': torch.tensor([6.0, 8.0])},
        ]
        
        weights = [0.3, 0.7]  # Weights
        result = aggregator.aggregate(gradients_per_gpu, gpu_weights=weights)
        
        # Verify results: 2*0.3 + 6*0.7 = 4.8, 4*0.3 + 8*0.7 = 6.8
        expected = torch.tensor([4.8, 6.8])
        self.assertTrue(torch.allclose(result['param1'], expected))
    
    def test_gradient_clipping(self):
        """Test gradient clipping"""
        aggregator = GradientAggregator(
            enable_gradient_clipping=True,
            max_grad_norm=1.0
        )
        
        # Create large gradients
        large_gradients = [
            {'param1': torch.tensor([10.0, 20.0])},  # Norm is large
        ]
        
        result = aggregator.aggregate(large_gradients)
        
        # Verify gradients are clipped
        result_norm = torch.norm(result['param1']).item()
        self.assertLessEqual(result_norm, 1.0)
    
    def test_consistency_validation(self):
        """Test consistency validation"""
        # Inconsistent gradient structure
        inconsistent_gradients = [
            {'param1': torch.tensor([1.0, 2.0])},
            {'param2': torch.tensor([3.0, 4.0])},  # Different parameter names
        ]
        
        # Should record consistency failures
        original_failures = self.aggregator.consistency_failures
        result = self.aggregator.aggregate(inconsistent_gradients)
        
        # Verify failure count increased
        self.assertGreater(self.aggregator.consistency_failures, original_failures)

class TestMultiGPUManager(unittest.TestCase):
    """Test MultiGPUManager component (simulated environment)"""
    
    def setUp(self):
        if not CORE_AVAILABLE:
            self.skipTest("Core components not available")
        
        # Create a simple model
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # Simulate testing in single GPU environment (avoid actual multi-GPU needs)
        self.num_gpus = 1
        
        self.optimizer_config = {
            'class': torch.optim.AdamW,
            'kwargs': {'lr': 0.001, 'weight_decay': 0.01}
        }
    
    def test_manager_initialization(self):
        """Test manager initialization"""
        manager = MultiGPUManager(
            self.model, 
            self.num_gpus, 
            self.optimizer_config
        )
        
        self.assertEqual(manager.num_gpus, self.num_gpus)
        self.assertEqual(len(manager.devices), self.num_gpus)
        self.assertFalse(manager._setup_complete)
    
    def test_gradient_aggregation_interface(self):
        """Test gradient aggregation interface"""
        manager = MultiGPUManager(
            self.model, 
            self.num_gpus, 
            self.optimizer_config
        )
        
        # Simulate gradients
        gradients_per_gpu = [
            {'weight': torch.tensor([1.0, 2.0]), 'bias': torch.tensor([0.5])},
        ]
        
        result = manager.aggregate_gradients(gradients_per_gpu)
        
        # In a single GPU scenario, the original gradients should be returned
        self.assertTrue(torch.equal(result['weight'], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(result['bias'], torch.tensor([0.5])))
    
    def test_performance_stats(self):
        """Test performance stats"""
        manager = MultiGPUManager(
            self.model, 
            self.num_gpus, 
            self.optimizer_config
        )
        
        stats = manager.get_performance_stats()
        
        self.assertIn('total_steps', stats)
        self.assertIn('num_gpus', stats)
        self.assertIn('setup_complete', stats)
        self.assertEqual(stats['num_gpus'], self.num_gpus)

class TestIntegration(unittest.TestCase):
    """Integration test"""
    
    def setUp(self):
        if not CORE_AVAILABLE:
            self.skipTest("Core components not available")
    
    def test_full_pipeline_simulation(self):
        """Test full pipeline simulation"""
        # Create components
        num_gpus = 2
        batch_sharder = BatchSharding(num_gpus)
        gradient_aggregator = GradientAggregator()
        
        # Simulate training batch
        batch_size = 8
        batch = {
            'input_ids': torch.randint(0, 100, (batch_size, 32)),
            'labels': torch.randint(0, 10, (batch_size,))
        }
        
        # 1. Shard batch
        shards = batch_sharder.split_batch(batch)
        self.assertEqual(len(shards), num_gpus)
        
        # 2. Simulate gradient calculation on each GPU
        gradients_per_gpu = []
        for i, shard in enumerate(shards):
            # Simulate gradient calculation
            grad_dict = {
                'layer1.weight': torch.randn(10, 5) * 0.01,
                'layer1.bias': torch.randn(10) * 0.01
            }
            gradients_per_gpu.append(grad_dict)
        
        # 3. Aggregate gradients
        aggregated_gradients = gradient_aggregator.aggregate(gradients_per_gpu)
        
        # Verify aggregation results
        self.assertIn('layer1.weight', aggregated_gradients)
        self.assertIn('layer1.bias', aggregated_gradients)
        
        # 4. Collect outputs (simulated loss)
        losses = [0.5, 0.6]  # Losses on each GPU
        avg_loss = batch_sharder.gather_outputs(losses)
        self.assertAlmostEqual(avg_loss, 0.55)

def run_tests():
    """Run all tests"""
    if not CORE_AVAILABLE:
        print("❌ Core components not available, skipping tests")
        return
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBatchSharding,
        TestGradientAggregator, 
        TestMultiGPUManager,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print results
    if result.wasSuccessful():
        print(f"\n✅ All tests passed! ({result.testsRun} tests)")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")
        
    return result.wasSuccessful()

def run_core_tests():
    """Run core component tests (alias function for unified interface)"""
    return run_tests()

if __name__ == '__main__':
    run_tests() 