"""
GPU Device Management Utility
Provides device detection, configuration, monitoring, and optimization features.
"""

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    GPU Device Manager

    Responsible for:
    1. GPU device detection and configuration
    2. Device status monitoring
    3. Device inter-communication configuration
    4. Device performance optimization
    """

    def __init__(self):
        """Initialize the device manager"""
        self.device_info = {}
        self.monitoring_active = False
        self.monitor_thread = None
        self.performance_stats = {}

        self._initialize_devices()
        logger.info("🔧 Device manager initialized")

    def _initialize_devices(self):
        """Initialize device information"""
        if not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not available, running in CPU mode")
            return

        device_count = torch.cuda.device_count()
        logger.info(f"🔍 Found {device_count} GPU device(s)")

        for i in range(device_count):
            device_info = self._get_device_properties(i)
            self.device_info[i] = device_info
            logger.info(f"📱 GPU {i}: {device_info['name']} ({device_info['memory_gb']:.1f} GB)")

    def _get_device_properties(self, device_id: int) -> Dict[str, Any]:
        """
        Get device properties

        Args:
            device_id: Device ID

        Returns:
            Dict: Device property information
        """
        try:
            props = torch.cuda.get_device_properties(device_id)

            return {
                'device_id': device_id,
                'name': props.name,
                'memory_gb': props.total_memory / 1024**3,
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessor_count': props.multi_processor_count,
                'max_threads_per_multiprocessor': props.max_threads_per_multi_processor,
                'max_shared_memory_per_multiprocessor': props.max_shared_memory_per_multi_processor,
                'memory_clock_rate': props.memory_clock_rate,
                'memory_bus_width': props.memory_bus_width,
                'l2_cache_size': props.l2_cache_size,
                'max_threads_per_block': props.max_threads_per_block,
                'is_integrated': props.integrated,
                'is_multi_gpu_board': props.multi_gpu_board
            }
        except Exception as e:
            logger.error(f"❌ Failed to get properties for device {device_id}: {e}")
            return {'error': str(e)}

    def get_available_devices(self) -> List[int]:
        """
        Get list of available GPU devices

        Returns:
            List: List of available device IDs
        """
        if not torch.cuda.is_available():
            return []

        available_devices = []
        for device_id in range(torch.cuda.device_count()):
            try:
                # Test if device is available
                with torch.cuda.device(device_id):
                    torch.cuda.current_device()
                available_devices.append(device_id)
            except Exception as e:
                logger.warning(f"⚠️ Device {device_id} is not available: {e}")

        return available_devices

    def get_optimal_device_assignment(self, num_gpus: int = None) -> List[int]:
        """
        Get optimal device assignment

        Args:
            num_gpus: Number of GPUs needed

        Returns:
            List: Recommended device ID list
        """
        available_devices = self.get_available_devices()

        if not available_devices:
            logger.warning("⚠️ No available GPU devices")
            return []

        if num_gpus is None:
            num_gpus = len(available_devices)

        # Sort by memory size
        device_scores = []
        for device_id in available_devices:
            info = self.device_info.get(device_id, {})
            if 'memory_gb' in info:
                score = info['memory_gb']
                # Consider compute capability
                if 'compute_capability' in info:
                    major, minor = map(int, info['compute_capability'].split('.'))
                    score += (major * 10 + minor) * 0.1
                device_scores.append((device_id, score))

        # Sort and select optimal devices
        device_scores.sort(key=lambda x: x[1], reverse=True)
        optimal_devices = [device_id for device_id, _ in device_scores[:num_gpus]]

        logger.info(f"🎯 Recommended devices: {optimal_devices}")
        return optimal_devices

    def check_device_compatibility(self, device_ids: List[int]) -> Dict[str, Any]:
        """
        Check device compatibility

        Args:
            device_ids: List of device IDs to check

        Returns:
            Dict: Compatibility check result
        """
        result = {
            'compatible': True,
            'warnings': [],
            'errors': [],
            'device_comparison': {}
        }

        if len(device_ids) < 2:
            return result

        # Get all device info
        devices_info = []
        for device_id in device_ids:
            info = self.device_info.get(device_id, {})
            if 'error' in info:
                result['errors'].append(f"Failed to get info for device {device_id}")
                result['compatible'] = False
                continue
            devices_info.append((device_id, info))

        if len(devices_info) < 2:
            return result

        # Check compute capability
        compute_capabilities = [info['compute_capability'] for _, info in devices_info]
        if len(set(compute_capabilities)) > 1:
            result['warnings'].append(f"Devices have different compute capabilities: {dict(zip(device_ids, compute_capabilities))}")

        # Check memory size
        memory_sizes = [info['memory_gb'] for _, info in devices_info]
        min_memory = min(memory_sizes)
        max_memory = max(memory_sizes)

        if max_memory - min_memory > max_memory * 0.2:  # Difference exceeds 20%
            result['warnings'].append(f"Devices have large memory size differences: {dict(zip(device_ids, memory_sizes))}")

        # Check device architecture
        architectures = [info['name'] for _, info in devices_info]
        if len(set(architectures)) > 1:
            result['warnings'].append(f"Devices have different architectures: {dict(zip(device_ids, architectures))}")

        # Generate comparison info
        for device_id, info in devices_info:
            result['device_comparison'][device_id] = {
                'name': info['name'],
                'memory_gb': info['memory_gb'],
                'compute_capability': info['compute_capability'],
                'multiprocessor_count': info['multiprocessor_count']
            }

        return result

    def configure_devices_for_multi_gpu(self, device_ids: List[int]) -> bool:
        """
        Configure devices for multi-GPU training

        Args:
            device_ids: List of device IDs

        Returns:
            bool: Whether configuration succeeded
        """
        try:
            if not device_ids:
                logger.error("❌ Device list is empty")
                return False

            # Check device compatibility
            compatibility = self.check_device_compatibility(device_ids)

            for warning in compatibility['warnings']:
                logger.warning(f"⚠️ {warning}")

            if compatibility['errors']:
                for error in compatibility['errors']:
                    logger.error(f"❌ {error}")
                return False

            # Set device environment variable
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))

            # Initialize devices
            for device_id in device_ids:
                try:
                    with torch.cuda.device(device_id):
                        # Warm up device
                        dummy = torch.zeros(1).cuda()
                        del dummy
                        torch.cuda.empty_cache()
                    logger.info(f"✅ Device {device_id} initialized successfully")
                except Exception as e:
                    logger.error(f"❌ Device {device_id} failed to initialize: {e}")
                    return False

            # Enable P2P communication (if supported)
            self._configure_p2p_communication(device_ids)

            logger.info(f"🚀 Multi-GPU configuration complete: {len(device_ids)} device(s)")
            return True

        except Exception as e:
            logger.error(f"❌ Multi-GPU device configuration failed: {e}")
            return False

    def _configure_p2p_communication(self, device_ids: List[int]):
        """Configure P2P communication"""
        try:
            p2p_matrix = {}

            for i, device_i in enumerate(device_ids):
                for j, device_j in enumerate(device_ids):
                    if i != j:
                        # Check P2P access capability
                        can_access = torch.cuda.can_device_access_peer(device_i, device_j)
                        p2p_matrix[(device_i, device_j)] = can_access

                        if can_access:
                            logger.info(f"🔗 Device {device_i} -> {device_j} supports P2P communication")
                        else:
                            logger.warning(f"⚠️ Device {device_i} -> {device_j} does not support P2P communication")

            # Store P2P matrix
            self.p2p_matrix = p2p_matrix

        except Exception as e:
            logger.warning(f"⚠️ Failed to configure P2P communication: {e}")

    def start_device_monitoring(self, interval: float = 1.0):
        """
        Start device monitoring

        Args:
            interval: Monitoring interval (seconds)
        """
        if self.monitoring_active:
            logger.warning("⚠️ Device monitoring is already running")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_devices,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"📊 Started device monitoring, interval: {interval}s")

    def stop_device_monitoring(self):
        """Stop device monitoring"""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)

        logger.info("🛑 Device monitoring stopped")

    def _monitor_devices(self, interval: float):
        """Device monitoring loop"""
        while self.monitoring_active:
            try:
                timestamp = time.time()
                stats = {}

                for device_id in self.get_available_devices():
                    with torch.cuda.device(device_id):
                        stats[device_id] = {
                            'memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                            'memory_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                            'utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0,
                            'temperature': self._get_gpu_temperature(device_id)
                        }

                self.performance_stats[timestamp] = stats

                # Only keep the most recent stats
                if len(self.performance_stats) > 1000:
                    oldest_key = min(self.performance_stats.keys())
                    del self.performance_stats[oldest_key]

                time.sleep(interval)

            except Exception as e:
                logger.error(f"❌ Device monitoring error: {e}")
                time.sleep(interval)

    def _get_gpu_temperature(self, device_id: int) -> Optional[float]:
        """Get GPU temperature (if supported)"""
        try:
            # Here you can integrate nvidia-ml-py to get temperature
            # Currently returns None as a placeholder
            return None
        except Exception:
            return None

    def get_device_stats(self, device_id: int = None) -> Dict[str, Any]:
        """
        Get device statistics

        Args:
            device_id: Device ID, None for all devices

        Returns:
            Dict: Device statistics
        """
        if not self.performance_stats:
            return {'error': 'Monitoring not started or no data'}

        latest_timestamp = max(self.performance_stats.keys())
        latest_stats = self.performance_stats[latest_timestamp]

        if device_id is not None:
            return latest_stats.get(device_id, {'error': f'No data for device {device_id}'})

        return latest_stats

    def optimize_device_settings(self, device_ids: List[int]) -> Dict[str, Any]:
        """
        Optimize device settings

        Args:
            device_ids: List of device IDs

        Returns:
            Dict: Optimization results
        """
        results = {
            'optimizations_applied': [],
            'performance_improvements': [],
            'warnings': []
        }

        try:
            for device_id in device_ids:
                with torch.cuda.device(device_id):
                    # Set best performance mode
                    torch.backends.cudnn.benchmark = True
                    results['optimizations_applied'].append(f'device_{device_id}_cudnn_benchmark')

                    # Enable mixed precision
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    results['optimizations_applied'].append(f'device_{device_id}_tf32_enabled')

                    logger.info(f"🎯 Device {device_id} optimization complete")

            results['performance_improvements'].append("Enabled cuDNN benchmark mode")
            results['performance_improvements'].append("Enabled TF32 mixed precision")

        except Exception as e:
            logger.error(f"❌ Device optimization failed: {e}")
            results['warnings'].append(str(e))

        return results

    def get_device_summary(self) -> Dict[str, Any]:
        """Get device summary information"""
        summary = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
            'devices': self.device_info,
            'monitoring_active': self.monitoring_active,
            'p2p_matrix': getattr(self, 'p2p_matrix', {}),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None
        }

        return summary

    def __del__(self):
        """Destructor, stops monitoring"""
        try:
            self.stop_device_monitoring()
        except Exception:
            pass


def get_optimal_gpu_configuration(num_gpus: int = None) -> Dict[str, Any]:
    """
    Convenience function to get optimal GPU configuration

    Args:
        num_gpus: Number of GPUs needed

    Returns:
        Dict: Optimal configuration information
    """
    device_manager = DeviceManager()

    config = {
        'device_manager': device_manager,
        'available_devices': device_manager.get_available_devices(),
        'optimal_assignment': device_manager.get_optimal_device_assignment(num_gpus),
        'device_summary': device_manager.get_device_summary()
    }

    # Check compatibility
    if config['optimal_assignment']:
        compatibility = device_manager.check_device_compatibility(config['optimal_assignment'])
        config['compatibility'] = compatibility

    return config


def configure_multi_gpu_environment(device_ids: List[int] = None) -> bool:
    """
    Convenience function to configure multi-GPU environment

    Args:
        device_ids: List of device IDs, None for auto selection

    Returns:
        bool: Whether configuration succeeded
    """
    device_manager = DeviceManager()

    if device_ids is None:
        device_ids = device_manager.get_optimal_device_assignment()

    if not device_ids:
        logger.error("❌ No available GPU devices")
        return False

    success = device_manager.configure_devices_for_multi_gpu(device_ids)

    if success:
        # Apply performance optimizations
        optimization_results = device_manager.optimize_device_settings(device_ids)
        logger.info(f"🚀 Multi-GPU environment configured successfully: {device_ids}")
        logger.info(f"🎯 Applied {len(optimization_results['optimizations_applied'])} optimizations")

    return success 