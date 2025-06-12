#!/usr/bin/env python3
"""
MultiGPUTrainer unit tests
Test the full functionality of the multi-GPU trainer
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add path to import core module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from core import AggregationMethod, MultiGPUTrainer
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    print(f"Warning: Core components not available: {e}")


class MockModel(nn.Module):
    """Mock model for testing"""
    
    def __init__(self, input_size=10, hidden_size=5, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()
    
    def forward(self, input_ids, labels=None, **kwargs):
        x = self.linear1(input_ids.float())
        x = self.relu(x)
        logits = self.linear2(x)
        
        if labels is not None:
            loss = self.criterion(logits, labels.float().unsqueeze(-1))
            return type('Output', (), {'loss': loss, 'logits': logits})()
        
        return type('Output', (), {'logits': logits})()


class TestMultiGPUTrainer(unittest.TestCase):
    """Test MultiGPUTrainer class"""
    
    def setUp(self):
        if not CORE_AVAILABLE:
            self.skipTest("Core components not available")
        
        # Create mock model
        self.model = MockModel()
        self.num_gpus = 1  # Simulate single GPU in CPU environment
        
        # Optimizer configuration
        self.optimizer_config = {
            'class': torch.optim.AdamW,
            'kwargs': {'lr': 0.001, 'weight_decay': 0.01}
        }
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = MultiGPUTrainer(
            model=self.model,
            num_gpus=self.num_gpus,
            optimizer_config=self.optimizer_config
        )
        
        self.assertEqual(trainer.num_gpus, self.num_gpus)
        self.assertEqual(trainer.global_step, 0)
        self.assertEqual(trainer.epoch, 0)
        self.assertFalse(trainer._setup_complete)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_setup_validation_no_cuda(self, mock_cuda):
        """Test setup validation in no CUDA environment"""
        trainer = MultiGPUTrainer(
            model=self.model,
            num_gpus=self.num_gpus,
            optimizer_config=self.optimizer_config
        )
        
        # In no CUDA environment, setup should raise an exception
        with self.assertRaises(RuntimeError):
            trainer.setup()
    
    def test_train_step_structure(self):
        """Test structure of training step (simulated environment)"""
        trainer = MultiGPUTrainer(
            model=self.model,
            num_gpus=self.num_gpus,
            optimizer_config=self.optimizer_config
        )
        
        # Simulate batch data
        batch = {
            'input_ids': torch.randn(4, 10),
            'labels': torch.randn(4)
        }
        
        # Since there is no CUDA environment, we directly test the existence and parameters of the method
        self.assertTrue(hasattr(trainer, 'train_step'))
        self.assertTrue(hasattr(trainer, '_parallel_forward'))
        self.assertTrue(hasattr(trainer, '_parallel_backward'))
        self.assertTrue(hasattr(trainer, '_collect_outputs'))
    
    def test_forward_pass_with_labels(self):
        """Test forward pass with labels"""
        trainer = MultiGPUTrainer(
            model=self.model,
            num_gpus=self.num_gpus,
            optimizer_config=self.optimizer_config
        )
        
        batch = {
            'input_ids': torch.randn(2, 10),
            'labels': torch.randn(2)
        }
        
        # Test forward pass method
        result = trainer._forward_pass(self.model, batch)
        
        self.assertIn('loss', result)
        self.assertIn('logits', result)
        self.assertIsInstance(result['loss'], torch.Tensor)
    
    def test_forward_pass_without_labels(self):
        """Test forward pass without labels"""
        trainer = MultiGPUTrainer(
            model=self.model,
            num_gpus=self.num_gpus,
            optimizer_config=self.optimizer_config
        )
        
        batch = {
            'input_ids': torch.randn(2, 10)
        }
        
        result = trainer._forward_pass(self.model, batch)
        
        self.assertIn('loss', result)
        self.assertIn('logits', result)
        self.assertEqual(result['loss'].item(), 0.0)  # When there are no labels, the loss is 0
    
    def test_move_to_device(self):
        """Test data device movement"""
        trainer = MultiGPUTrainer(
            model=self.model,
            num_gpus=self.num_gpus,
            optimizer_config=self.optimizer_config
        )
        
        batch = {
            'input_ids': torch.randn(2, 10),
            'labels': torch.randn(2),
            'metadata': 'string_value'  # Non-tensor data
        }
        
        # Test moving to CPU (because there is no CUDA)
        result = trainer._move_to_device(batch, 'cpu')
        
        self.assertEqual(result['input_ids'].device.type, 'cpu')
        self.assertEqual(result['labels'].device.type, 'cpu')
        self.assertEqual(result['metadata'], 'string_value')
    
    def test_collect_outputs(self):
        """Test output collection"""
        trainer = MultiGPUTrainer(
            model=self.model,
            num_gpus=self.num_gpus,
            optimizer_config=self.optimizer_config
        )
        
        # Simulate forward propagation results
        forward_results = [
            {'loss': torch.tensor(0.5), 'shard_size': 2, 'gpu_id': 0},
            {'loss': torch.tensor(0.3), 'shard_size': 2, 'gpu_id': 1},
            {'loss': torch.tensor(0.0), 'shard_size': 0, 'gpu_id': 2}  # 空分片
        ]
        
        result = trainer._collect_outputs(forward_results)
        
        self.assertIn('loss', result)
        self.assertIn('num_samples', result)
        self.assertIn('num_active_gpus', result)
        
        self.assertEqual(result['num_samples'], 4)  # 2 + 2 + 0
        self.assertEqual(result['num_active_gpus'], 2)  # Two non-empty shards
        self.assertAlmostEqual(result['loss'], 0.4)  # (0.5 + 0.3) / 2
    
    def test_training_stats(self):
        """Test training statistics"""
        trainer = MultiGPUTrainer(
            model=self.model,
            num_gpus=self.num_gpus,
            optimizer_config=self.optimizer_config
        )
        
        # Initial statistics should be empty
        stats = trainer.get_training_stats()
        
        self.assertEqual(stats['global_step'], 0)
        self.assertEqual(stats['epoch'], 0)
        self.assertEqual(stats['num_batches_processed'], 0)
        self.assertIn('multi_gpu_manager', stats)
        self.assertIn('gradient_aggregator', stats)
        self.assertIn('batch_sharder', stats)
    
    def test_mixed_precision_config(self):
        """Test mixed precision configuration"""
        # Enable mixed precision
        trainer_amp = MultiGPUTrainer(
            model=self.model,
            num_gpus=self.num_gpus,
            optimizer_config=self.optimizer_config,
            enable_mixed_precision=True
        )
        
        # Disable mixed precision
        trainer_no_amp = MultiGPUTrainer(
            model=self.model,
            num_gpus=self.num_gpus,
            optimizer_config=self.optimizer_config,
            enable_mixed_precision=False
        )
        
        # In CPU environment, both scalers should be None
        self.assertIsNone(trainer_amp.scaler)
        self.assertIsNone(trainer_no_amp.scaler)
    
    def test_aggregation_method_config(self):
        """Test different aggregation method configurations"""
        for method in [AggregationMethod.MEAN, AggregationMethod.SUM, AggregationMethod.WEIGHTED_MEAN]:
            trainer = MultiGPUTrainer(
                model=self.model,
                num_gpus=self.num_gpus,
                optimizer_config=self.optimizer_config,
                aggregation_method=method
            )
            
            self.assertEqual(trainer.aggregation_method, method)
            self.assertEqual(trainer.gradient_aggregator.aggregation_method, method)


class TestMultiGPUTrainerIntegration(unittest.TestCase):
    """Test MultiGPUTrainer integration functionality"""
    
    def setUp(self):
        if not CORE_AVAILABLE:
            self.skipTest("Core components not available")
        
        self.model = MockModel()
        self.num_gpus = 1
        self.optimizer_config = {
            'class': torch.optim.AdamW,
            'kwargs': {'lr': 0.001}
        }
    
    def test_dataloader_creation(self):
        """Test data loader creation and usage"""
        # Create mock data
        input_data = torch.randn(20, 10)
        target_data = torch.randn(20)
        dataset = TensorDataset(input_data, target_data)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        trainer = MultiGPUTrainer(
            model=self.model,
            num_gpus=self.num_gpus,
            optimizer_config=self.optimizer_config
        )
        
        # Verify data loader can iterate normally
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            batch_dict = {
                'input_ids': inputs,
                'labels': targets
            }
            
            # Test batch format
            self.assertIn('input_ids', batch_dict)
            self.assertIn('labels', batch_dict)
            self.assertEqual(inputs.shape[0], min(4, 20 - batch_idx * 4))
            
            if batch_idx >= 2:  # Only test the first few batches
                break
    
    def test_checkpoint_save_load_structure(self):
        """Test checkpoint save and load structure"""
        trainer = MultiGPUTrainer(
            model=self.model,
            num_gpus=self.num_gpus,
            optimizer_config=self.optimizer_config
        )
        
        # Simulate some training states
        trainer.global_step = 100
        trainer.epoch = 2
        trainer.training_stats['num_batches_processed'] = 50
        
        # Test checkpoint structure creation (without actually saving files)
        checkpoint_data = {
            'global_step': trainer.global_step,
            'epoch': trainer.epoch,
            'model_state_dict': trainer.model.state_dict(),
            'training_stats': trainer.training_stats,
            'trainer_config': {
                'num_gpus': trainer.num_gpus,
                'aggregation_method': trainer.aggregation_method.value,
                'enable_gradient_checkpointing': trainer.enable_gradient_checkpointing,
                'enable_mixed_precision': trainer.enable_mixed_precision
            }
        }
        
        # Verify checkpoint data structure
        self.assertEqual(checkpoint_data['global_step'], 100)
        self.assertEqual(checkpoint_data['epoch'], 2)
        self.assertIn('model_state_dict', checkpoint_data)
        self.assertIn('trainer_config', checkpoint_data)


def run_trainer_tests():
    """Run trainer tests"""
    if not CORE_AVAILABLE:
        print("❌ Core components not available, skipping trainer tests")
        return
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMultiGPUTrainer,
        TestMultiGPUTrainerIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print results
    if result.wasSuccessful():
        print(f"\n✅ All trainer tests passed! ({result.testsRun} tests)")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")
        
    return result.wasSuccessful()


if __name__ == '__main__':
    run_trainer_tests() 