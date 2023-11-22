"""
.. admonition:: Monitoring
    :class: hint

    현재 기기의 상태를 살펴볼 수 있는 함수들
"""

from zerohertzLib.monitoring.cpu import cpu
from zerohertzLib.monitoring.gpu import gpu_memory, gpu_usages
from zerohertzLib.monitoring.storage import storage

__all__ = ["storage", "cpu", "gpu_memory", "gpu_usages"]
