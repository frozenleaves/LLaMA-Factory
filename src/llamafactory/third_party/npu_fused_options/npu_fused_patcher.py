from types import ModuleType

from . import sdpa_attention as npu_sdpa_attention
from . import rms_norm, rope, swiglu


def _patch_sdpa_forward():
    """
    The purpose of this patch is to enable the native SDPA forward function of transformers to adapt to the
    SDPA interface of NPU. If not, calling the SDPA interface is still in the eagle mode
    """
    from transformers.integrations import sdpa_attention
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

    sdpa_attention.sdpa_attention_forward = npu_sdpa_attention.sdpa_attention_forward
    AttentionInterface._global_mapping["sdpa"] = npu_sdpa_attention.sdpa_attention_forward
    ALL_ATTENTION_FUNCTIONS["sdpa"] = npu_sdpa_attention.sdpa_attention_forward


def _patch_rmsnorm(module: ModuleType, class_name: str):

    setattr(module, class_name, rms_norm.NpuRMSNorm)


def _patch_rope(module: ModuleType, func_name: str):
    setattr(module, func_name, rope.apply_rotary_pos_emb)


def _patch_swiglu(module: ModuleType, class_name: str):
    setattr(module, class_name, swiglu.NpuSwiGlu)


def apply_fused_options(disable: bool=False):
    if disable:
        return
    from transformers.models.qwen2 import modeling_qwen2
    from transformers.models.qwen2_moe import modeling_qwen2_moe
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.qwen3_moe import modeling_qwen3_moe

    _patch_sdpa_forward()
    _patch_rmsnorm(modeling_qwen2, "Qwen2RMSNorm")
    _patch_rope(modeling_qwen2, "apply_rotary_pos_emb")
    _patch_swiglu(modeling_qwen2, "Qwen2MLP")




