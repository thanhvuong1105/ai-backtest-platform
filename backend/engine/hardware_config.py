# engine/hardware_config.py
"""
Hardware-aware configuration for high-performance systems.

Optimized for:
- CPU: Intel i9-12900K (8P + 8E = 24 threads)
- GPU: RTX 3080 (10GB VRAM)
- RAM: 64GB DDR5

Usage:
    from .hardware_config import get_config, detect_hardware
    config = get_config()
"""

import os
import multiprocessing
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# HARDWARE DETECTION
# ═══════════════════════════════════════════════════════

def detect_hardware() -> Dict[str, Any]:
    """Detect available hardware and return capabilities."""
    import platform

    cpu_count = os.cpu_count() or 4

    # Detect GPU
    gpu_available = False
    gpu_name = None
    gpu_memory = 0

    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    except ImportError:
        pass

    # Detect RAM (approximate)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total // (1024**3)
    except ImportError:
        ram_gb = 16  # Default assumption

    # Detect if Apple Silicon
    is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'

    return {
        'cpu_count': cpu_count,
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'gpu_memory_gb': gpu_memory,
        'ram_gb': ram_gb,
        'is_apple_silicon': is_apple_silicon,
        'platform': platform.system()
    }


# ═══════════════════════════════════════════════════════
# PERFORMANCE PROFILES
# ═══════════════════════════════════════════════════════

# Profile: Apple Silicon (M1/M2/M3) - Memory efficient
PROFILE_APPLE_SILICON = {
    'name': 'apple_silicon',
    'cpu': {
        'n_workers': 8,
        'chunk_size': 100_000,
        'use_numba': True,
        'process_pool': False,  # Use ThreadPool for memory efficiency
    },
    'gpu': {
        'enabled': False,  # MPS not optimized for this workload
    },
    'ram': {
        'data_cache_size': 64,
        'indicator_cache_size': 256,
        'max_memory_gb': 8,
    },
    'optimization': {
        'population_size': 50,
        'generations': 10,
        'parallel_fitness': True,
        'batch_size': 10,
    }
}

# Profile: Standard Desktop (8-core, no GPU, 16-32GB RAM)
PROFILE_STANDARD = {
    'name': 'standard',
    'cpu': {
        'n_workers': 8,
        'chunk_size': 200_000,
        'use_numba': True,
        'process_pool': False,
    },
    'gpu': {
        'enabled': False,
    },
    'ram': {
        'data_cache_size': 128,
        'indicator_cache_size': 512,
        'max_memory_gb': 16,
    },
    'optimization': {
        'population_size': 100,
        'generations': 20,
        'parallel_fitness': True,
        'batch_size': 16,
    }
}

# Profile: High Performance (i9/Ryzen 9, RTX 3080+, 64GB+ RAM)
PROFILE_HIGH_PERFORMANCE = {
    'name': 'high_performance',
    'cpu': {
        'n_workers': 16,
        'chunk_size': 500_000,
        'use_numba': True,
        'process_pool': True,  # Use ProcessPool for CPU-bound work
    },
    'gpu': {
        'enabled': True,
        'device': 'cuda:0',
        'batch_size': 8192,
        'mixed_precision': True,
        'memory_limit_gb': 8,
    },
    'ram': {
        'data_cache_size': 256,
        'indicator_cache_size': 2048,
        'max_memory_gb': 48,
    },
    'optimization': {
        'population_size': 500,
        'generations': 100,
        'parallel_fitness': True,
        'batch_size': 32,
    }
}

# Profile: Server/Cloud (many cores, optional GPU, lots of RAM)
PROFILE_SERVER = {
    'name': 'server',
    'cpu': {
        'n_workers': 32,
        'chunk_size': 1_000_000,
        'use_numba': True,
        'process_pool': True,
    },
    'gpu': {
        'enabled': True,
        'device': 'cuda:0',
        'batch_size': 16384,
        'mixed_precision': True,
        'memory_limit_gb': 16,
    },
    'ram': {
        'data_cache_size': 512,
        'indicator_cache_size': 4096,
        'max_memory_gb': 128,
    },
    'optimization': {
        'population_size': 2000,
        'generations': 500,
        'parallel_fitness': True,
        'batch_size': 64,
    }
}

# Profile: EXTREME - Maximum settings for your specific hardware (i9-12900K, RTX 3080, 64GB)
PROFILE_EXTREME = {
    'name': 'extreme',
    'cpu': {
        'n_workers': 16,  # 8P-cores + reserve for E-cores
        'chunk_size': 500_000,
        'use_numba': True,
        'process_pool': True,
    },
    'gpu': {
        'enabled': True,
        'device': 'cuda:0',
        'batch_size': 8192,
        'mixed_precision': True,
        'memory_limit_gb': 8,  # Reserve 2GB for display
    },
    'ram': {
        'data_cache_size': 512,
        'indicator_cache_size': 4096,
        'max_memory_gb': 56,  # Reserve 8GB for system
    },
    'optimization': {
        'population_size': 2000,
        'generations': 500,
        'parallel_fitness': True,
        'batch_size': 50,
        'elite_count': 20,
        'mutation_rate': 0.15,
        'crossover_rate': 0.8,
        'tournament_size': 7,
    },
    'robustness': {
        'monte_carlo_runs': 1000,
        'walk_forward_windows': 5,
        'stress_scenarios': 6,
    }
}


# ═══════════════════════════════════════════════════════
# AUTO-DETECTION AND CONFIGURATION
# ═══════════════════════════════════════════════════════

def select_profile(hardware: Dict[str, Any]) -> Dict[str, Any]:
    """Select the best profile based on detected hardware."""

    # Check for explicit override
    profile_override = os.getenv('QUANT_BRAIN_PROFILE', '').lower()
    if profile_override == 'extreme':
        return PROFILE_EXTREME
    elif profile_override == 'high_performance':
        return PROFILE_HIGH_PERFORMANCE
    elif profile_override == 'server':
        return PROFILE_SERVER
    elif profile_override == 'standard':
        return PROFILE_STANDARD
    elif profile_override == 'apple_silicon':
        return PROFILE_APPLE_SILICON

    # Auto-detect based on hardware
    if hardware['is_apple_silicon']:
        logger.info("Detected Apple Silicon - using memory-efficient profile")
        return PROFILE_APPLE_SILICON

    cpu_count = hardware['cpu_count']
    ram_gb = hardware['ram_gb']
    gpu_available = hardware['gpu_available']
    gpu_memory = hardware['gpu_memory_gb']

    # High-end system: 12+ cores, 48GB+ RAM, GPU with 8GB+ VRAM
    if cpu_count >= 12 and ram_gb >= 48 and gpu_available and gpu_memory >= 8:
        logger.info(f"Detected high-end system ({cpu_count} cores, {ram_gb}GB RAM, {gpu_memory}GB VRAM) - using HIGH_PERFORMANCE profile")
        return PROFILE_HIGH_PERFORMANCE

    # Server: 24+ cores, 64GB+ RAM
    if cpu_count >= 24 and ram_gb >= 64:
        logger.info(f"Detected server ({cpu_count} cores, {ram_gb}GB RAM) - using SERVER profile")
        return PROFILE_SERVER

    # Standard system
    logger.info(f"Using STANDARD profile ({cpu_count} cores, {ram_gb}GB RAM)")
    return PROFILE_STANDARD


def get_config() -> Dict[str, Any]:
    """
    Get the optimal configuration for the current hardware.

    Returns a dictionary with optimized settings for CPU, GPU, RAM, and optimization parameters.
    """
    hardware = detect_hardware()
    profile = select_profile(hardware)

    # Apply environment variable overrides
    config = apply_env_overrides(profile.copy())

    # Log configuration
    logger.info(f"Hardware Config: {profile['name']} profile")
    logger.info(f"  CPU Workers: {config['cpu']['n_workers']}")
    logger.info(f"  GPU Enabled: {config['gpu']['enabled']}")
    logger.info(f"  Population: {config['optimization']['population_size']}")
    logger.info(f"  Generations: {config['optimization']['generations']}")

    return config


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration."""

    # CPU overrides
    if os.getenv('MAX_THREAD_WORKERS'):
        config['cpu']['n_workers'] = int(os.getenv('MAX_THREAD_WORKERS'))

    # Optimization overrides
    if os.getenv('GENOME_POPULATION_SIZE'):
        config['optimization']['population_size'] = int(os.getenv('GENOME_POPULATION_SIZE'))
    if os.getenv('GENOME_GENERATIONS'):
        config['optimization']['generations'] = int(os.getenv('GENOME_GENERATIONS'))
    if os.getenv('GENOME_MUTATION_RATE'):
        config['optimization']['mutation_rate'] = float(os.getenv('GENOME_MUTATION_RATE'))
    if os.getenv('GENOME_CROSSOVER_RATE'):
        config['optimization']['crossover_rate'] = float(os.getenv('GENOME_CROSSOVER_RATE'))

    # Cache overrides
    if os.getenv('INDICATOR_CACHE_SIZE'):
        config['ram']['indicator_cache_size'] = int(os.getenv('INDICATOR_CACHE_SIZE'))
    if os.getenv('DATA_CACHE_SIZE'):
        config['ram']['data_cache_size'] = int(os.getenv('DATA_CACHE_SIZE'))

    # GPU override
    if os.getenv('DISABLE_GPU', 'false').lower() == 'true':
        config['gpu']['enabled'] = False

    return config


# ═══════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════

def get_worker_count() -> int:
    """Get optimal number of workers for parallel processing."""
    config = get_config()
    return config['cpu']['n_workers']


def get_population_size() -> int:
    """Get optimal population size for genetic algorithm."""
    config = get_config()
    return config['optimization']['population_size']


def get_generations() -> int:
    """Get optimal number of generations for genetic algorithm."""
    config = get_config()
    return config['optimization']['generations']


def is_gpu_enabled() -> bool:
    """Check if GPU acceleration should be used."""
    config = get_config()
    return config['gpu']['enabled']


def get_batch_size() -> int:
    """Get optimal batch size for parallel processing."""
    config = get_config()
    return config['optimization']['batch_size']


# ═══════════════════════════════════════════════════════
# PERFORMANCE MONITORING
# ═══════════════════════════════════════════════════════

def print_hardware_info():
    """Print detected hardware information."""
    hardware = detect_hardware()
    profile = select_profile(hardware)

    print("\n" + "="*60)
    print("QUANT BRAIN - Hardware Configuration")
    print("="*60)
    print(f"Platform:       {hardware['platform']}")
    print(f"CPU Cores:      {hardware['cpu_count']}")
    print(f"RAM:            {hardware['ram_gb']} GB")
    print(f"GPU:            {hardware['gpu_name'] or 'Not detected'}")
    if hardware['gpu_available']:
        print(f"GPU Memory:     {hardware['gpu_memory_gb']} GB")
    print(f"Apple Silicon:  {hardware['is_apple_silicon']}")
    print("-"*60)
    print(f"Selected Profile: {profile['name'].upper()}")
    print(f"  Workers:      {profile['cpu']['n_workers']}")
    print(f"  Population:   {profile['optimization']['population_size']}")
    print(f"  Generations:  {profile['optimization']['generations']}")
    print(f"  GPU Enabled:  {profile['gpu']['enabled']}")
    print("="*60 + "\n")


# Run on import if DEBUG
if os.getenv('QUANT_BRAIN_DEBUG', 'false').lower() == 'true':
    print_hardware_info()
