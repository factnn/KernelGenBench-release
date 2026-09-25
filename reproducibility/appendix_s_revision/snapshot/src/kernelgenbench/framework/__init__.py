"""
Framework module

Provides base interfaces for framework adapters and argument generation.
"""

from .adapter import FrameworkAdapter
from .generate_args import BaseGenerateArgs, TritonKernelGenerateArgs, InputArg, OutputArg
from .torch_adapter import TorchAdapter

__all__ = [
    'FrameworkAdapter',
    'BaseGenerateArgs',
    'TritonKernelGenerateArgs',
    'InputArg',
    'OutputArg',
    'TorchAdapter',
]
