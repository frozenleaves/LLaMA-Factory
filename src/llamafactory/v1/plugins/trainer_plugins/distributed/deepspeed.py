# Copyright 2025 the LlamaFactory team.
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

import copy
import json
import os
from functools import partialmethod
from typing import Any, Dict, Optional

import torch

try:
    import deepspeed
except ImportError as e:
    raise RuntimeError("DeepSpeed is not installed. Please install it to use this plugin.") from e

from ....accelerator.interface import DistributedInterface
from ....utils.logging import get_logger


logger = get_logger(__name__)


class DeepSpeedConfigHelper:
    """
    Helper class to process DeepSpeed configuration and sync with training arguments.
    Simplified version of HfDeepSpeedConfig/HfTrainerDeepSpeedConfig.
    """

    def __init__(self, config_file_or_dict):
        if isinstance(config_file_or_dict, dict):
            self.config = copy.deepcopy(config_file_or_dict)
        elif isinstance(config_file_or_dict, str) and os.path.exists(config_file_or_dict):
            with open(config_file_or_dict, encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Expected a string path to an existing deepspeed config or a dictionary. Received: {config_file_or_dict}")

        self.mismatches = []

    def _find_config_node(self, ds_key_long):
        config = self.config
        nodes = ds_key_long.split(".")
        ds_key = nodes.pop()
        for node in nodes:
            config = config.get(node)
            if config is None:
                return None, ds_key
        return config, ds_key

    def get_value(self, ds_key_long, default=None):
        config, ds_key = self._find_config_node(ds_key_long)
        if config is None:
            return default
        return config.get(ds_key, default)

    def _fill_match(self, ds_key_long, hf_val, hf_key=None, must_match=True):
        config, ds_key = self._find_config_node(ds_key_long)
        if config is None:
            return

        if config.get(ds_key) == "auto":
            config[ds_key] = hf_val
            return

        if not must_match:
            return

        ds_val = config.get(ds_key)
        if ds_val is not None and ds_val != hf_val:
            self.mismatches.append(f"- ds {ds_key_long}={ds_val} vs hf {hf_key}={hf_val}")

    _fill_only = partialmethod(_fill_match, must_match=False)

    def process_config(self, args, auto_find_batch_size=False):
        """
        Sync config with training arguments.
        """
        train_batch_size = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._fill_match("train_micro_batch_size_per_gpu", args.per_device_train_batch_size, "per_device_train_batch_size", not auto_find_batch_size)
        self._fill_match("gradient_accumulation_steps", args.gradient_accumulation_steps, "gradient_accumulation_steps")
        self._fill_match("train_batch_size", train_batch_size, "train_batch_size (calculated)", not auto_find_batch_size)
        self._fill_match("gradient_clipping", args.max_grad_norm, "max_grad_norm")

        self._fill_match("optimizer.params.lr", args.learning_rate, "learning_rate")
        self._fill_match("optimizer.params.betas", [args.adam_beta1, args.adam_beta2], "adam_beta1+adam_beta2")
        self._fill_match("optimizer.params.eps", args.adam_epsilon, "adam_epsilon")
        self._fill_match("optimizer.params.weight_decay", args.weight_decay, "weight_decay")

        self._fill_only("scheduler.params.warmup_min_lr", 0)
        self._fill_match("scheduler.params.warmup_max_lr", args.learning_rate, "learning_rate")

        if args.save_on_each_node:
            self.config.setdefault("checkpoint", {})["use_node_local_storage"] = args.save_on_each_node

        self._fill_match("fp16.enabled", (args.fp16 or args.fp16_full_eval), "fp16|fp16_full_eval")
        self._fill_match("bf16.enabled", (args.bf16 or args.bf16_full_eval), "bf16|bf16_full_eval")

    def finalize_config(self, args, model, num_training_steps):
        """
        Finalize config with model info and training steps.
        """
        # zero optimization auto config based on hidden size
        hidden_size_based_keys = [
            "zero_optimization.reduce_bucket_size",
            "zero_optimization.stage3_prefetch_bucket_size",
            "zero_optimization.stage3_param_persistence_threshold",
        ]
        
        if any(self.get_value(k) == "auto" for k in hidden_size_based_keys):
            hidden_size = None
            if hasattr(model, "config"):
                if hasattr(model.config, "hidden_size"):
                    hidden_size = model.config.hidden_size
                elif hasattr(model.config, "hidden_sizes"):
                    hidden_size = max(model.config.hidden_sizes)
                elif hasattr(model.config, "text_config"):
                    if hasattr(model.config.text_config, "hidden_size"):
                        hidden_size = model.config.text_config.hidden_size
                    elif hasattr(model.config.text_config, "hidden_sizes"):
                        hidden_size = max(model.config.text_config.hidden_sizes)

            if hidden_size is None:
                raise ValueError("Could not determine hidden_size for auto DeepSpeed config.")

            self._fill_only("zero_optimization.reduce_bucket_size", hidden_size * hidden_size)
            if self.get_value("zero_optimization.stage") == 3:
                self._fill_only("zero_optimization.stage3_prefetch_bucket_size", int(0.9 * hidden_size * hidden_size))
                self._fill_only("zero_optimization.stage3_param_persistence_threshold", 10 * hidden_size)

        self._fill_match("scheduler.params.total_num_steps", num_training_steps, "num_training_steps (calculated)")
        self._fill_match("scheduler.params.warmup_num_steps", args.get_warmup_steps(num_training_steps), "warmup_steps")

        if self.mismatches:
            raise ValueError("DeepSpeed config mismatch:\n" + "\n".join(self.mismatches))


class TrainingArgsWrapper:
    def __init__(self, batch_config):
        self.batch_config = batch_config
        self.world_size = batch_config.get("dp_size", 1)
        self.per_device_train_batch_size = batch_config.get("micro_batch_size", 1)
        self.gradient_accumulation_steps = batch_config.get("num_micro_batch", 1)
        self.max_grad_norm = batch_config.get("max_grad_norm", 1.0)
        self.learning_rate = batch_config.get("learning_rate", 1e-5)
        self.weight_decay = batch_config.get("weight_decay", 0.0)
        self.adam_beta1 = batch_config.get("adam_beta1", 0.9)
        self.adam_beta2 = batch_config.get("adam_beta2", 0.999)
        self.adam_epsilon = batch_config.get("adam_epsilon", 1e-8)
        self.save_on_each_node = False
        self.fp16 = not batch_config.get("bf16", False)
        self.bf16 = batch_config.get("bf16", False)
        self.fp16_full_eval = self.fp16
        self.bf16_full_eval = self.bf16

    def get_warmup_steps(self, num_training_steps):
        warmup_ratio = self.batch_config.get("warmup_ratio", 0.0)
        return int(num_training_steps * warmup_ratio)


class DeepSpeedEngine:
    def __init__(self, dist_config: Dict[str, Any], batch_config: Optional[Dict[str, Any]] = None):
        self.dist_config = dist_config
        self.batch_config = batch_config or {}

        config_file = dist_config.get("config_file", None)
        if not config_file:
            raise ValueError("DeepSpeed config_file is required")
        
        self.config_helper = DeepSpeedConfigHelper(config_file)
        self.args_wrapper = TrainingArgsWrapper(self.batch_config)

        # Process config with args
        self.config_helper.process_config(self.args_wrapper)
        
        # Expose the config dict for compatibility
        self.ds_config = self.config_helper.config

    def shard_model(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        # Finalize config with model and num_training_steps
        num_training_steps = self.batch_config.get("num_training_steps", 1)
        self.config_helper.finalize_config(self.args_wrapper, model, num_training_steps)
        
        # If optimizer is configured in DeepSpeed config, do not pass the external optimizer
        if "optimizer" in self.ds_config:
            logger.info_rank0("Optimizer is defined in DeepSpeed config. Ignoring the external optimizer.")
            optimizer = None

        init_result = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=self.ds_config,
        )
        
        logger.info_rank0(f"DeepSpeed initialized with config: {json.dumps(self.ds_config, indent=2)}")
        
        # Unpack result
        # engine, optimizer, dataloader, lr_scheduler
        engine = init_result[0]
        ds_optimizer = init_result[1] if len(init_result) > 1 else None
        ds_lr_scheduler = init_result[2] if len(init_result) > 2 else None

        # Attach these to the engine for easy access in hooks
        engine._ds_optimizer = ds_optimizer
        engine._ds_lr_scheduler = ds_lr_scheduler
        
        return engine


def post_shard(trainer) -> None:
    ds_engine = trainer.model
    # Update trainer's optimizer/scheduler references if DeepSpeed created them
    ds_optimizer = getattr(ds_engine, "optimizer", None) or getattr(ds_engine, "_ds_optimizer", None)
    if ds_optimizer is not None:
        trainer.optimizer = ds_optimizer

    ds_lr_scheduler = getattr(ds_engine, "lr_scheduler", None) or getattr(ds_engine, "_ds_lr_scheduler", None)
    if ds_lr_scheduler is not None:
        trainer.lr_scheduler = ds_lr_scheduler


class DeepSpeedCompatibleLRScheduler:
    """
    A learning rate scheduler that is compatible with DeepSpeed optimizer.
    This scheduler maintains its own state without directly interacting with the DeepSpeed optimizer.
    """
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.last_epoch = -1
        self.base_lr = trainer.args.learning_rate
        self.current_lr = self.base_lr
        
        # Initialize internal scheduler based on lr_scheduler_config
        if trainer.args.lr_scheduler_config is not None:
            self._init_internal_scheduler()
        else:
            self.internal_scheduler = None
    
    def _init_internal_scheduler(self):
        """Initialize an internal scheduler for tracking learning rate changes."""
        try:
            from ..lr_scheduler import LRSchedulerPlugin
            
            # Create a dummy optimizer for the internal scheduler
            # This is just for calculating learning rate values, not for actual optimization
            dummy_param = torch.nn.Parameter(torch.tensor(1.0))
            dummy_optimizer = torch.optim.AdamW([dummy_param], lr=self.base_lr)
            
            self.internal_scheduler = LRSchedulerPlugin(self.trainer.args.lr_scheduler_config.name)(
                dummy_optimizer, self.trainer.num_training_steps, self.trainer.args.lr_scheduler_config
            )
        except Exception as e:
            logger.warning_rank0(f"Failed to create internal scheduler: {e}, using constant LR")
            self.internal_scheduler = None
    
    def step(self):
        """Update scheduler state. Called by BaseTrainer."""
        self.last_epoch += 1
        if self.internal_scheduler is not None:
            self.internal_scheduler.step()
            self.current_lr = self.internal_scheduler.get_last_lr()[0]
            
            # Apply to DeepSpeed optimizer
            ds_engine = self.trainer.model
            ds_optimizer = getattr(ds_engine, "optimizer", None) or getattr(ds_engine, "_ds_optimizer", None)
            if ds_optimizer is not None:
                for param_group in ds_optimizer.param_groups:
                    param_group['lr'] = self.current_lr
    
    def state_dict(self):
        """Return scheduler state for checkpointing."""
        state = {
            'last_epoch': self.last_epoch,
            'current_lr': self.current_lr,
        }
        if self.internal_scheduler is not None:
            state['internal_scheduler'] = self.internal_scheduler.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.last_epoch = state_dict.get('last_epoch', -1)
        self.current_lr = state_dict.get('current_lr', self.base_lr)
        if 'internal_scheduler' in state_dict and self.internal_scheduler is not None:
            self.internal_scheduler.load_state_dict(state_dict['internal_scheduler'])
    
    def get_last_lr(self):
        """Return current learning rate."""
        if self.internal_scheduler is not None:
            return self.internal_scheduler.get_last_lr()
        return [self.current_lr]


def init_lr_scheduler(trainer):
    """
    Get the LR scheduler from DeepSpeed engine.
    
    Returns:
        The DeepSpeed scheduler if it was created by DeepSpeed initialization,
        or a DeepSpeed-compatible scheduler based on trainer's lr_scheduler_config.
    """
    ds_engine = trainer.model
    ds_lr_scheduler = getattr(ds_engine, "lr_scheduler", None) or getattr(ds_engine, "_ds_lr_scheduler", None)

    if ds_lr_scheduler is not None:
        logger.info_rank0("Using DeepSpeed internal scheduler.")
        return ds_lr_scheduler

    # If DS didn't create a scheduler (i.e., no "scheduler" in JSON config),
    # create a DeepSpeed-compatible scheduler based on lr_scheduler_config.
    logger.info_rank0("No DeepSpeed scheduler found. Creating DeepSpeed-compatible scheduler based on lr_scheduler_config.")
    
    if trainer.args.lr_scheduler_config is not None:
        logger.info_rank0(f"Using lr_scheduler_config: {trainer.args.lr_scheduler_config.name}")
    else:
        logger.info_rank0("No lr_scheduler_config specified, using constant learning rate")
    
    return DeepSpeedCompatibleLRScheduler(trainer)


def backward(trainer, loss) -> None:
    ds_engine = trainer.model
    ds_engine.backward(loss)
    ds_engine.step()


def opt_step(trainer) -> float:
    ds_engine = trainer.model
    return ds_engine.get_global_grad_norm() or 0.0


def save_model(trainer) -> None:
    """Save the model."""    

    ds_engine = trainer.model
    rank = DistributedInterface().get_rank()
    # Check for ZeRO-3
    is_zero3 = False
    if hasattr(ds_engine, "zero_optimization_stage"):
        stage = ds_engine.zero_optimization_stage()
        is_zero3 = (stage == 3)

    output_dir = trainer.args.output_dir
        
    if is_zero3:
        # Unwrap the model to get the HF model
        model_to_save = ds_engine.module
        
        # Ensure we are saving the state dict properly
        # deepspeed.zero.GatheredParameters is needed for getting the weights
        with deepspeed.zero.GatheredParameters(list(model_to_save.parameters()), modifier_rank=0):
            if rank == 0:
                # Move state_dict to CPU to avoid OOM during saving
                state_dict = model_to_save.state_dict()
                cpu_state_dict = {k: v.cpu()for k, v in state_dict.items()}
                del state_dict # Free GPU memory
                
                model_to_save.save_pretrained(output_dir, state_dict=cpu_state_dict, max_shard_size="5GB")
                trainer.renderer.processor.save_pretrained(output_dir)
                del cpu_state_dict # Free CPU memory
    else:
        # For ZeRO-1/2 or no ZeRO, rank 0 can save.
        if rank == 0:
            model_to_save = ds_engine.module
            
            # Move state_dict to CPU to avoid OOM during saving
            state_dict = model_to_save.state_dict()
            cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
            del state_dict # Free GPU memory
            
            model_to_save.save_pretrained(output_dir, state_dict=cpu_state_dict, max_shard_size="4GB")
            trainer.renderer.processor.save_pretrained(output_dir)
            del cpu_state_dict # Free CPU memory

    DistributedInterface().barrier()
    logger.info_rank0(f"Model saved to {output_dir}")
