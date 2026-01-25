# Copyright 2025 the LlamaFactory team.
#
# This code is inspired by the MindSpeed library.
# https://gitee.com/ascend/MindSpeed/blob/master/mindspeed/lite/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FSDP2 + Expert Parallelism (EP) plugin for distributed training.

This module implements a distributed training strategy that combines:
- FSDP2 (Fully Sharded Data Parallel v2) for sharding non-expert parameters
- EP (Expert Parallelism) for distributing MoE experts across ranks
- EFSDP (Expert FSDP) for additional sharding of expert parameters

The key design principles:
1. Independent device mesh for EP (edp, efsdp, ep) to avoid conflicts with standard FSDP
2. Two-stage sharding for experts: EP on dim=0, EFSDP on dim=1
3. Proper gradient scaling across all parallel dimensions
"""

import fnmatch
import gc
import os
import types
from typing import Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DTensor, Shard, distribute_tensor
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from transformers import PreTrainedModel

from ....accelerator.helper import get_current_accelerator
from ....accelerator.interface import DistributedInterface
from ....utils.logging import get_logger
from .ep_utils.dispatcher import get_experts_forward_fn
from .ep_utils.parallel_state import ExpertParallelState


logger = get_logger(__name__)


def module_name_match(pattern: str, name: str) -> bool:
    """Check if module name matches the pattern.

    Supports wildcards like "*.mlp.experts" or "model.layers.*.mlp.experts".

    Args:
        pattern: Pattern to match (supports * and ? wildcards).
        name: Module name to check.

    Returns:
        True if the name matches the pattern.
    """
    return fnmatch.fnmatch(name, pattern)


def get_transformer_layer_cls(model: PreTrainedModel) -> Optional[type[nn.Module]]:
    """Get the transformer layer class from a model.

    Args:
        model: The HuggingFace pretrained model.

    Returns:
        The transformer layer class or None if not found.
    """
    no_split_modules = getattr(model, "_no_split_modules", None)
    if no_split_modules:
        if isinstance(no_split_modules, (list, tuple)):
            for name, module in model.named_modules():
                for cls_name in no_split_modules:
                    if module.__class__.__name__ == cls_name:
                        return module.__class__

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return type(model.model.layers[0])
    if hasattr(model, "layers"):
        return type(model.layers[0])

    return None


class FSDP2EPEngine:
    """Engine for FSDP2 + Expert Parallelism distributed training.

    This engine applies:
    1. Expert Parallelism (EP): Distributes experts across EP ranks
    2. Expert FSDP (EFSDP): Additional FSDP sharding for expert parameters
    3. Standard FSDP: For non-expert model parameters

    Args:
        dist_config: Distribution configuration dictionary containing:
            - ep_size: Expert parallel size (default: 1)
            - efsdp_size: Expert FSDP size (default: 1)
            - mixed_precision: Mixed precision policy ("bf16", "fp16", "fp32")
            - reshard_after_forward: Whether to reshard after forward pass
            - offload_params: Whether to offload parameters to CPU
            - pin_memory: Whether to pin memory for CPU offload
            - dcp_path: Path for distributed checkpoint
            - expert_modules: List of patterns to match expert modules
    """

    def __init__(self, dist_config: dict) -> None:
        self.dist_interface = DistributedInterface()
        self.rank = self.dist_interface.get_rank()
        self.local_rank = self.dist_interface.get_local_rank()
        self.world_size = self.dist_interface.get_world_size()

        # EP configuration
        self.ep_size = dist_config.get("ep_size", 1)
        self.efsdp_size = dist_config.get("efsdp_size", 1)
        self.expert_module_patterns = dist_config.get("expert_modules", [])

        # FSDP configuration
        self.mixed_precision = dist_config.get("mixed_precision", "bf16")
        self.reshard_after_forward = dist_config.get("reshard_after_forward", True)
        self.offload_params = dist_config.get("offload_params", False)
        self.pin_memory = dist_config.get("pin_memory", True)
        self.dcp_path = dist_config.get("dcp_path", None)

        # Initialize EP parallel state if EP is enabled
        if self.ep_size > 1 or self.efsdp_size > 1:
            self.ep_state = ExpertParallelState(
                ep_size=self.ep_size,
                efsdp_size=self.efsdp_size,
            )
        else:
            self.ep_state = None

        # Get standard FSDP device mesh from DistributedInterface
        self.device_mesh = self.dist_interface.data_device_mesh
        if self.device_mesh is not None:
            try:
                self.fsdp_mesh = self.device_mesh["dp"]
            except Exception:
                self.fsdp_mesh = self.device_mesh
            logger.info(f"Using FSDP Device Mesh: {self.fsdp_mesh}")
        else:
            self.fsdp_mesh = None
            logger.warning("Device Mesh not found. FSDP2 might fail if not running in distributed mode.")

    def get_mp_policy(self) -> MixedPrecisionPolicy:
        """Get mixed precision policy based on configuration.

        Returns:
            MixedPrecisionPolicy for FSDP.
        """
        if self.mixed_precision == "bf16":
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
        elif self.mixed_precision == "fp16":
            param_dtype = torch.float16
            reduce_dtype = torch.float32
        else:
            param_dtype = torch.float32
            reduce_dtype = torch.float32

        return MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=True,
        )

    def _get_expert_modules(self, model: PreTrainedModel) -> list[tuple[str, nn.Module]]:
        """Find all expert modules matching the configured patterns.

        Args:
            model: The model to search.

        Returns:
            List of (name, module) tuples for matched expert modules.
        """
        expert_modules = []
        for pattern in self.expert_module_patterns:
            for name, module in model.named_modules():
                if module_name_match(pattern, name):
                    expert_modules.append((name, module))
                    if self.rank == 0:
                        logger.info(f"Found expert module matching pattern '{pattern}': {name}")

        return expert_modules

    def _get_expert_param_names(self, model: PreTrainedModel) -> Set[str]:
        """Get parameter names that belong to expert modules.

        Args:
            model: The model.

        Returns:
            Set of fully qualified parameter names belonging to expert modules.
        """
        expert_param_names = set()
        for name, module in self._get_expert_modules(model):
            for param_name, _ in module.named_parameters():
                full_name = f"{name}.{param_name}" if name else param_name
                expert_param_names.add(full_name)
        return expert_param_names

    def _apply_expert_parallel(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply Expert Parallelism to expert modules.

        This distributes expert weights across EP ranks and replaces the forward
        function with one that includes AllToAll communication.

        Args:
            model: The model to modify.

        Returns:
            Model with EP applied to expert modules.
        """
        if self.ep_state is None or not self.ep_state.is_ep_enabled():
            return model

        ep_mesh = self.ep_state.ep_mesh
        ep_group = self.ep_state.ep_group
        ep_rank = self.ep_state.ep_rank
        ep_size = self.ep_size

        for module_name, module in self._get_expert_modules(model):
            # Determine number of experts
            if isinstance(module, nn.ModuleList):
                num_global_experts = len(module)
            elif hasattr(module, "num_experts"):
                num_global_experts = module.num_experts
            else:
                logger.warning(f"Cannot determine number of experts for module {module_name}, skipping EP.")
                continue

            if num_global_experts % ep_size != 0:
                raise ValueError(
                    f"Number of experts ({num_global_experts}) must be divisible by ep_size ({ep_size})."
                )

            num_local_experts = num_global_experts // ep_size
            local_expert_offset = ep_rank * num_local_experts
            local_expert_indices = [local_expert_offset + i for i in range(num_local_experts)]

            # Store EP metadata on module
            module.num_global_experts = num_global_experts
            module.num_local_experts = num_local_experts
            module.local_expert_indices = local_expert_indices

            if self.rank == 0:
                logger.info(
                    f"Applying EP to {module_name}: "
                    f"{num_global_experts} global experts, {num_local_experts} local experts per rank"
                )

            # Distribute expert weights using DTensor with Shard(0)
            self._distribute_expert_weights(module, ep_mesh)

            # Replace forward function with EP-aware version
            new_forward = get_experts_forward_fn(
                ep_group=ep_group,
                num_global_experts=num_global_experts,
                num_local_experts=num_local_experts,
                ep_rank=ep_rank,
            )
            module.forward = types.MethodType(new_forward, module)

            # Apply gradient division hook for proper averaging
            self._apply_grad_division_hook(module, ep_size)

        return model

    def _distribute_expert_weights(self, module: nn.Module, ep_mesh: DeviceMesh) -> None:
        """Distribute expert weights as DTensors sharded on dim=0.

        Args:
            module: Expert module.
            ep_mesh: Device mesh for EP dimension.
        """
        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                # Shard on first dimension (expert dimension)
                dist_param = nn.Parameter(distribute_tensor(param.data, ep_mesh, [Shard(0)]))
                module.register_parameter(name, dist_param)

        for child_name, child_module in module.named_children():
            self._distribute_expert_weights(child_module, ep_mesh)

    def _apply_grad_division_hook(self, module: nn.Module, divide_factor: float) -> None:
        """Apply gradient division hooks for proper gradient averaging.

        Args:
            module: Module to apply hooks to.
            divide_factor: Factor to divide gradients by.
        """
        for param in module.parameters():
            if param.requires_grad:
                # Use gradient accumulation hook
                param.register_post_accumulate_grad_hook(
                    lambda p, factor=divide_factor: p.grad.mul_(1.0 / factor)
                )

    def _apply_expert_fsdp(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply Expert FSDP to expert modules.

        This adds additional FSDP sharding on dim=1 for expert parameters
        that have already been EP-sharded on dim=0.

        Args:
            model: The model to modify.

        Returns:
            Model with EFSDP applied to expert modules.
        """
        if self.ep_state is None or not self.ep_state.is_efsdp_enabled():
            return model

        efsdp_mesh = self.ep_state.efsdp_mesh
        mp_policy = self.get_mp_policy()

        # Configuration for expert FSDP: shard on dim=1 since dim=0 is already EP-sharded
        efsdp_config = {
            "mesh": efsdp_mesh,
            "mp_policy": mp_policy,
            "shard_placement_fn": lambda x: Shard(1),  # Shard on second dimension
        }

        for module_name, module in self._get_expert_modules(model):
            if isinstance(module, nn.ModuleList):
                for i, expert in enumerate(module):
                    fully_shard(expert, **efsdp_config)
                    if self.rank == 0:
                        logger.info(f"Applied EFSDP to {module_name}[{i}]")
            else:
                fully_shard(module, **efsdp_config)
                if self.rank == 0:
                    logger.info(f"Applied EFSDP to {module_name}")

            # Update gradient divide factor to include EFSDP
            if hasattr(module, "set_gradient_divide_factor"):
                module.set_gradient_divide_factor(self.ep_state.gradient_divide_factor)

        return model

    def _apply_standard_fsdp(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply standard FSDP to non-expert model parameters.

        Args:
            model: The model to modify.

        Returns:
            Model with FSDP applied to non-expert parameters.
        """
        if self.fsdp_mesh is None:
            logger.warning("No FSDP Mesh available, skipping standard FSDP wrapping.")
            return model

        mp_policy = self.get_mp_policy()
        layer_cls = get_transformer_layer_cls(model)

        if layer_cls is None:
            logger.warning("Could not identify Transformer Layer class, applying FSDP to model root only.")
            transformer_layer_cls_to_wrap = set()
        else:
            logger.info(f"Applying per-layer FSDP to {layer_cls.__name__}")
            transformer_layer_cls_to_wrap = {layer_cls}

        # Get expert parameter names to exclude from standard FSDP
        expert_param_names = self._get_expert_param_names(model)

        # Collect parameters to ignore (expert parameters)
        ignored_params = set()
        for name, param in model.named_parameters():
            if name in expert_param_names:
                ignored_params.add(param)

        # Apply FSDP to transformer layers (excluding expert modules)
        for name, module in model.named_modules():
            should_wrap = False

            # Check if this is a transformer layer
            if type(module) in transformer_layer_cls_to_wrap:
                should_wrap = True
            elif isinstance(module, nn.Embedding):
                if not getattr(model.config, "tie_word_embeddings", True):
                    should_wrap = True

            # Skip if this module is an expert module
            is_expert = any(
                module_name_match(pattern, name) for pattern in self.expert_module_patterns
            )
            if is_expert:
                should_wrap = False

            if should_wrap:
                # Get module's own ignored params
                module_ignored_params = set(p for p in module.parameters() if p in ignored_params)

                fully_shard(
                    module,
                    mesh=self.fsdp_mesh,
                    reshard_after_forward=self.reshard_after_forward,
                    mp_policy=mp_policy,
                    offload_policy=CPUOffloadPolicy(pin_memory=self.pin_memory) if self.offload_params else None,
                    ignored_params=module_ignored_params if module_ignored_params else None,
                )

        # Enable gradient checkpointing
        if self.rank == 0:
            logger.info("Enabling gradient checkpointing...")

        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # Apply FSDP to the model root
        fully_shard(
            model,
            mesh=self.fsdp_mesh,
            reshard_after_forward=self.reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=CPUOffloadPolicy(pin_memory=self.pin_memory) if self.offload_params else None,
            ignored_params=ignored_params if ignored_params else None,
        )

        return model

    def prepare_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Prepare model with EP, EFSDP, and standard FSDP.

        Args:
            model: The model to prepare.

        Returns:
            Prepared model with all parallel strategies applied.
        """
        # Step 1: Apply Expert Parallelism
        if self.ep_size > 1:
            model = self._apply_expert_parallel(model)

        # Step 2: Apply Expert FSDP
        if self.efsdp_size > 1:
            model = self._apply_expert_fsdp(model)

        # Step 3: Apply standard FSDP to non-expert parameters
        model = self._apply_standard_fsdp(model)

        return model

    @torch.no_grad()
    def materialize_and_load(
        self, model: PreTrainedModel, hf_model_path: str, dcp_path: Optional[str] = None
    ) -> PreTrainedModel:
        """Materialize sharded model parameters and load weights.

        Args:
            model: Model with meta-device parameters.
            hf_model_path: Path to HuggingFace model weights.
            dcp_path: Optional path to distributed checkpoint.

        Returns:
            Model with loaded weights.
        """
        if self.rank == 0:
            logger.info("Materializing sharded model params...")

        device = get_current_accelerator()
        model.to_empty(device=device)

        if dcp_path and os.path.exists(dcp_path):
            if self.rank == 0:
                logger.info(f"DCP path found at {dcp_path}. Using efficient Sharded Loading (DCP Load).")
            self._load_from_dcp(model, dcp_path)
        else:
            if self.rank == 0:
                if dcp_path:
                    logger.warning(f"DCP path {dcp_path} not found.")
                logger.info("Using HF Meta Loading (Chunk Load).")
            self._load_weights_from_hf_checkpoint(model, hf_model_path)

        return model

    def shard_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Main entry point for model sharding.

        Args:
            model: The model to shard.

        Returns:
            Sharded model ready for training.
        """
        if model.device.type == "meta":
            model = self.prepare_model(model)
            model = self.materialize_and_load(
                model, hf_model_path=model.config.name_or_path, dcp_path=self.dcp_path
            )
        else:
            model = self.prepare_model(model)
        return model

    def _load_from_dcp(self, model: PreTrainedModel, dcp_path: str) -> None:
        """Load model weights from distributed checkpoint.

        Args:
            model: The model to load weights into.
            dcp_path: Path to the distributed checkpoint.
        """
        import torch.distributed.checkpoint as dcp

        try:
            if self.rank == 0:
                logger.info(f"Loading distributed checkpoint from {dcp_path} ...")

            options = StateDictOptions(full_state_dict=False, cpu_offload=True)
            local_state_dict = get_model_state_dict(model, options=options)
            dcp.load(state_dict=local_state_dict, checkpoint_id=dcp_path)
            set_model_state_dict(model, local_state_dict, options=options)

            if self.rank == 0:
                logger.info("DCP weights loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load from DCP: {e}")
            raise e

    def _load_weights_from_hf_checkpoint(self, model: PreTrainedModel, hf_model_path: str) -> None:
        """Load model weights from HuggingFace checkpoint.

        Args:
            model: The model to load weights into.
            hf_model_path: Path to the HuggingFace model.
        """
        import glob
        import json

        hf_model_path = self._resolve_hf_checkpoint_dir(hf_model_path)

        if self.rank == 0:
            logger.info(f"Loading weights from {hf_model_path} ...")

        index_file = os.path.join(hf_model_path, "model.safetensors.index.json")
        is_safetensors = True
        checkpoint_files = []

        if os.path.exists(index_file):
            with open(index_file) as f:
                index = json.load(f)
            checkpoint_files = sorted(set(index["weight_map"].values()))
            checkpoint_files = [os.path.join(hf_model_path, f) for f in checkpoint_files]
        elif os.path.exists(os.path.join(hf_model_path, "model.safetensors")):
            checkpoint_files = [os.path.join(hf_model_path, "model.safetensors")]
        else:
            is_safetensors = False
            index_file = os.path.join(hf_model_path, "pytorch_model.bin.index.json")
            if os.path.exists(index_file):
                with open(index_file) as f:
                    index = json.load(f)
                checkpoint_files = sorted(set(index["weight_map"].values()))
                checkpoint_files = [os.path.join(hf_model_path, f) for f in checkpoint_files]
            elif os.path.exists(os.path.join(hf_model_path, "pytorch_model.bin")):
                checkpoint_files = [os.path.join(hf_model_path, "pytorch_model.bin")]
            else:
                checkpoint_files = sorted(glob.glob(os.path.join(hf_model_path, "*.safetensors")))
                if checkpoint_files:
                    is_safetensors = True
                else:
                    checkpoint_files = sorted(glob.glob(os.path.join(hf_model_path, "*.bin")))

        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {hf_model_path}")

        param_map = dict(model.named_parameters())
        total_files = len(checkpoint_files)

        for i, ckpt_file in enumerate(checkpoint_files):
            if self.rank == 0:
                logger.info(f"[{i + 1}/{total_files}] Loading {os.path.basename(ckpt_file)} ...")

            if is_safetensors:
                from safetensors import safe_open

                with safe_open(ckpt_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key in param_map:
                            tensor = f.get_tensor(key)
                            self._copy_weights(param_map[key], tensor)
            else:
                state_dict = torch.load(ckpt_file, map_location="cpu")
                for key, tensor in state_dict.items():
                    if key in param_map:
                        self._copy_weights(param_map[key], tensor)
                del state_dict
                gc.collect()

    def _resolve_hf_checkpoint_dir(self, hf_model_path: str) -> str:
        """Resolve HuggingFace model path to local directory.

        Args:
            hf_model_path: Model path or HuggingFace Hub ID.

        Returns:
            Local directory containing checkpoint files.
        """
        if not hf_model_path:
            return hf_model_path

        if os.path.isdir(hf_model_path):
            return hf_model_path
        if os.path.isfile(hf_model_path):
            return os.path.dirname(hf_model_path)

        # HuggingFace Hub repo id: snapshot to local cache
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ValueError(
                f"hf_model_path='{hf_model_path}' does not exist locally and huggingface_hub is not available. "
                f"Please provide a local model directory or install huggingface_hub. Error: {e}"
            ) from e

        revision = os.getenv("HF_REVISION")
        offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"

        allow_patterns = [
            "*.safetensors",
            "*.bin",
            "*.index.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "config.json",
        ]

        # In distributed runs, let rank0 download first
        if dist.is_available() and dist.is_initialized():
            if self.rank == 0:
                local_dir = snapshot_download(
                    repo_id=hf_model_path,
                    revision=revision,
                    local_files_only=offline,
                    allow_patterns=allow_patterns,
                )
                logger.info(f"Resolved HF repo id '{hf_model_path}' to local dir: {local_dir}")
            dist.barrier()
            if self.rank != 0:
                local_dir = snapshot_download(
                    repo_id=hf_model_path,
                    revision=revision,
                    local_files_only=True,
                    allow_patterns=allow_patterns,
                )
            return local_dir

        local_dir = snapshot_download(
            repo_id=hf_model_path,
            revision=revision,
            local_files_only=offline,
            allow_patterns=allow_patterns,
        )
        if self.rank == 0:
            logger.info(f"Resolved HF repo id '{hf_model_path}' to local dir: {local_dir}")
        return local_dir

    def _copy_weights(self, param: nn.Parameter, loaded_tensor: torch.Tensor) -> None:
        """Copy loaded tensor to parameter, handling DTensor sharding.

        Args:
            param: Target parameter (may be DTensor).
            loaded_tensor: Source tensor to copy from.
        """
        if loaded_tensor.dtype != param.dtype:
            loaded_tensor = loaded_tensor.to(param.dtype)

        if isinstance(param, DTensor):
            shard_placement = None
            mesh_dim = -1

            for i, placement in enumerate(param.placements):
                if isinstance(placement, Shard):
                    shard_placement = placement
                    mesh_dim = i
                    break

            local_tensor = param.to_local()

            if shard_placement is None:
                local_tensor.copy_(loaded_tensor)
            else:
                dim = shard_placement.dim
                mesh = param.device_mesh
                my_coordinate = mesh.get_coordinate()
                if my_coordinate is None:
                    return

                rank_in_dim = my_coordinate[mesh_dim]
                world_size_in_dim = mesh.size(mesh_dim)

                full_size = param.shape[dim]
                chunk_size = (full_size + world_size_in_dim - 1) // world_size_in_dim

                start = rank_in_dim * chunk_size
                end = min(start + chunk_size, full_size)

                if start >= full_size:
                    return

                sliced_tensor = loaded_tensor.narrow(dim, start, end - start)

                slices = [slice(None)] * local_tensor.ndim
                slices[dim] = slice(0, sliced_tensor.shape[dim])
                local_tensor[tuple(slices)].copy_(sliced_tensor)
        else:
            param.data.copy_(loaded_tensor)
