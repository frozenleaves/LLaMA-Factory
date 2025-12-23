from ....utils.plugin import BasePlugin
from .bootstrap import load_all_kernels


# 对接Base Plugin，处理分发接口

# 传参
# 多个接口给用户提供
# 默认，提供apply all，默认禁用
# 用户筛选， 通过二级参数传入指定kernel， apply_kernel调用

# 默认/指定检查失败来报错


available_kernels = load_all_kernels()


class KernelPlugin(BasePlugin):
    pass


@KernelPlugin("apply_kernel").register
def apply_kernel(kernel_id: str, **kwargs):
    kernel = available_kernels.get(kernel_id)
    if kernel is None:
        raise ValueError(f"Kernel {kernel_id} not found")
    return kernel.apply(**kwargs)


@KernelPlugin("apply_available_kernels").register
def apply_available_kernels(exclude_kernels: list[str] = None, **kwargs):
    for kernel in available_kernels:
        if kernel in exclude_kernels:
            continue
        apply_kernel(kernel, **kwargs)


@KernelPlugin("apply_outside_kernels").register
def apply_outside_kernels(**kwargs):
    pass
