from abc import ABC, abstractmethod
from typing import Any

from ....accelerator.helper import DeviceType, get_current_accelerator, is_torch_cuda_available, is_torch_npu_available
from ....utils.types import HFModel


class BaseKernel(ABC):
    # 子类需覆盖以下属性
    _kernel_id: Any = ""  # kernel ID, 可以是任意可 hash 值，标识一个kernel的具体实现
    _device: DeviceType = DeviceType.CPU  # "cuda", "npu", "cpu", et.

    @classmethod
    def get_kernel_id(cls) -> str:
        return cls._kernel_id

    @classmethod
    def get_device(cls) -> str:
        return cls._device

    @classmethod
    def check_deps(cls) -> bool:
        """检查依赖.

        在显式指定模式下，如果用户指定了某个实现但此检查失败，应报错而不是静默切换.
        kernel可重写此方法，以控制自己的需求依赖检查.
        """
        if cls._device != get_current_accelerator().type:
            return False
        if cls._device == DeviceType.NPU:
            return is_torch_npu_available()
        if cls._device == DeviceType.CUDA:
            return is_torch_cuda_available()
        return True

    @classmethod
    @abstractmethod
    def apply(cls, **kwargs) -> HFModel:
        """应用Kernel到模型上."""
        if not cls.check_deps():
            raise RuntimeError(f"{cls.__name__} is not available but {cls.__name__} kernel was called.")
        raise NotImplementedError
