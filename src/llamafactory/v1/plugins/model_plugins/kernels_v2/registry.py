from collections import defaultdict
from typing import Optional

from ....accelerator.helper import get_current_accelerator
from .base import BaseKernel


__all__ = ["Registry", "register_kernel"]


class Registry:
    # 存储结构: { "kernel_type": { "impl_id": Class } }
    # 例如: { "fused_moe": { "npu": NpuFusedMoEKernel, "cuda": CudaFusedMoEKernel }, "rope": { "npu": NpuRoPEKernel } }
    _kernels: dict[str, type[BaseKernel]] = defaultdict(dict)

    @classmethod
    def register(cls, kernel_cls: type[BaseKernel]):
        """注册装饰器.

        传入的类须指定 _kernel_type 和 _impl_id两个类属性，继承自BaseKernel的类包含这两个类属性.
        """
        if not issubclass(kernel_cls, BaseKernel):
            raise TypeError(f"Class {kernel_cls} must inherit from BaseKernel")
        kernel_id = kernel_cls.get_kernel_id()
        device = kernel_cls.get_device()

        # 运行时的设备类型与kernel要求的设备类型不一致，跳过注册
        if device != get_current_accelerator().type:
            return

        if not kernel_id:
            raise ValueError(f"Implementation ID (_impl_id) not defined for {kernel_cls}")

        # 检查重复注册
        if kernel_id in cls._kernels:
            raise ValueError(f"{kernel_id} already registered!, the registered kernel is {cls._kernels[kernel_id]}")

        cls._kernels[kernel_id] = kernel_cls
        return kernel_cls

    @classmethod
    def get(cls, kernel_id: str) -> Optional[type[BaseKernel]]:
        """获取指定的 Kernel 实现."""
        return cls._kernels.get(kernel_id)

    @classmethod
    def get_registered_kernels(cls) -> dict[str, type[BaseKernel]]:
        return cls._kernels


# 导出装饰器别名
register_kernel = Registry.register
