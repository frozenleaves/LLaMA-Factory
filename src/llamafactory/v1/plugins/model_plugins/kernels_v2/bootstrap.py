import importlib
from pathlib import Path

from ....utils.logging import get_logger
from .registry import Registry


logger = get_logger(__name__)


def load_all_kernels():
    """扫描 ops 目录下的所有 .py 文件并 import 它们.

    Import 动作会触发 @register_kernel 装饰器，完成自动注册.
    """
    # 定位到 ops 目录
    # 注意：这里假设 bootstrap.py 在 kernels_v2 根目录下
    ops_path = Path(__file__).parent / "ops"

    if not ops_path.exists():
        return

    # 获取当前包名 (例如 src.llamafactory...kernels_v2)
    base_package = __package__

    # 递归遍历 ops 目录
    for file_path in ops_path.rglob("*.py"):
        if file_path.name == "__init__.py":
            continue

        # 计算相对路径，例如:
        # file_path = .../kernels_v2/ops/mlp/npu_swiglu.py
        # rel_path  = ops/mlp/npu_swiglu.py
        rel_path = file_path.relative_to(Path(__file__).parent)

        # 构建 module path: ..kernels_v2.ops.mlp.npu_swiglu
        # list(parts) -> ['ops', 'mlp', 'npu_swiglu.py']
        # .join -> "ops.mlp.npu_swiglu.py"
        # [:-3] -> "ops.mlp.npu_swiglu"
        module_name = ".".join(rel_path.parts)[:-3]
        full_module_name = f"{base_package}.{module_name}"

        try:
            importlib.import_module(full_module_name)
        except Exception as e:
            # 即使某个文件 import 失败（例如缺少依赖），也不应该卡死整个加载过程
            # 具体的依赖检查会由 Registry.get 中的 check_deps 二次检查
            logger.warning(f"[Kernel Registry] Failed to import {full_module_name} when loading kernels: {e}")

    return Registry.get_registered_kernels()
