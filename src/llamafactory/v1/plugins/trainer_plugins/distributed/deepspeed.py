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

import json
import os
from typing import Any, Dict, Optional

import torch

try:
    import deepspeed
except ImportError as e:
    raise RuntimeError("DeepSpeed is not installed. Please install it to use this plugin.") from e

from ....accelerator.interface import DistributedInterface
from ....utils.logging import get_logger


logger = get_logger(__name__)

class DeepSpeedEngine:
    def __init__(self, dist_config: Dict[str, Any], batch_config: Optional[Dict[str, Any]] = None):
        self.dist_config = dist_config
        self.batch_config = batch_config or {}
        self.ds_config = self._load_config(dist_config)
    
    def _load_config(self, dist_config: Dict[str, Any]) -> Dict[str, Any]:
        config_file = dist_config.get("config_file", None)
        if not config_file:
            raise ValueError("DeepSpeed config_file is required")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"DeepSpeed config_file not found: {config_file}")
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Resolve "auto" values based on batch_config
        config = self._resolve_auto_config(config, self.batch_config)

        # Force gradient_accumulation_steps to 1 because LlamaFactory handles accumulation manually
        # in the training loop (iterating over micro-batches).
        # If we let DeepSpeed handle it, we would need to change the loop structure significantly.
        gas = config.get("gradient_accumulation_steps", None)
        if gas is not None and gas != 1:
            logger.warning_rank0(
                f"DeepSpeed gradient_accumulation_steps is set to {gas}. "
                "Overriding to 1 because the trainer handles gradient accumulation manually."
            )
        config["gradient_accumulation_steps"] = 1

        return config

    def _resolve_auto_config(self, config: Dict[str, Any], batch_config: Dict[str, Any]) -> Dict[str, Any]:
        def resolve(value, fallback):
            return fallback if value == "auto" else value

        micro_batch_size = batch_config.get("micro_batch_size", 1)
        dp_size = batch_config.get("dp_size", 1)
        max_grad_norm = batch_config.get("max_grad_norm", 1.0)
        bf16 = batch_config.get("bf16", False)

        # DeepSpeed expects these keys for auto configuration
        config["train_micro_batch_size_per_gpu"] = resolve(
            config.get("train_micro_batch_size_per_gpu", "auto"), micro_batch_size
        )
        
        config["train_batch_size"] = resolve(
            config.get("train_batch_size", "auto"),
            micro_batch_size * dp_size
        )

        config["gradient_clipping"] = resolve(config.get("gradient_clipping", "auto"), max_grad_norm)

        if "bf16" in config and isinstance(config["bf16"], dict):
            config["bf16"]["enabled"] = resolve(config["bf16"].get("enabled", "auto"), bool(bf16))
        if "fp16" in config and isinstance(config["fp16"], dict):
            config["fp16"]["enabled"] = resolve(config["fp16"].get("enabled", "auto"), bool(not bf16))

        # Resolve scheduler params
        if "scheduler" in config and isinstance(config["scheduler"], dict):
            scheduler_params = config["scheduler"].get("params", {})
            if "total_num_steps" in scheduler_params:
                scheduler_params["total_num_steps"] = resolve(
                    scheduler_params["total_num_steps"], batch_config.get("num_training_steps", 0)
                )
            
            if "warmup_num_steps" in scheduler_params:
                # Try to calculate warmup steps from ratio if available, otherwise use 0
                warmup_ratio = batch_config.get("warmup_ratio", 0.0)
                total_steps = scheduler_params.get("total_num_steps", 0)
                if total_steps != "auto":
                    warmup_steps = int(total_steps * warmup_ratio)
                else:
                    warmup_steps = 0
                
                scheduler_params["warmup_num_steps"] = resolve(
                    scheduler_params["warmup_num_steps"], warmup_steps
                )

            if "warmup_max_lr" in scheduler_params:
                scheduler_params["warmup_max_lr"] = resolve(
                    scheduler_params["warmup_max_lr"], batch_config.get("learning_rate", 0.0)
                )
            
            if "warmup_min_lr" in scheduler_params:
                scheduler_params["warmup_min_lr"] = resolve(
                    scheduler_params["warmup_min_lr"], 0.0
                )

        return config

    def shard_model(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        # If optimizer is configured in DeepSpeed config, do not pass the external optimizer
        if "optimizer" in self.ds_config:
            logger.info_rank0("Optimizer is defined in DeepSpeed config. Ignoring the external optimizer.")
            optimizer = None

        init_result = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=self.ds_config,
        )
        
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
            dummy_optimizer = torch.optim.SGD([dummy_param], lr=self.base_lr)
            
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
    trainer.model.backward(loss)


def opt_step(trainer) -> float:
    ds_engine = trainer.model
    
    ds_engine.step()
    
    ds_lr_scheduler = getattr(ds_engine, "lr_scheduler", None) or getattr(ds_engine, "_ds_lr_scheduler", None)
    if trainer.lr_scheduler is not None and trainer.lr_scheduler is not ds_lr_scheduler:
        trainer.lr_scheduler.step()
    
    return ds_engine.get_global_grad_norm()


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
        # For ZeRO-3, we need to gather parameters
        # We use the state_dict approach or save_pretrained if the model supports it under ZeRO-3
        # HuggingFace models often handle this if `save_pretrained` is called, but with DS engine wrapping it...
        
        # Best practice for ZeRO-3:
        # 1. Use `ds_engine.module.save_pretrained` but context managed.
        
        # Unwrap the model to get the HF model
        model_to_save = ds_engine.module
        
        # Ensure we are saving the state dict properly
        # deepspeed.zero.GatheredParameters is needed for getting the weights
        
        with deepspeed.zero.GatheredParameters(list(model_to_save.parameters()), modifier_rank=0):
            if rank == 0:
                model_to_save.save_pretrained(output_dir)
                trainer.renderer.processor.save_pretrained(output_dir)
    else:
        # For ZeRO-1/2 or no ZeRO, rank 0 can save.
        if rank == 0:
            model_to_save = ds_engine.module
            model_to_save.save_pretrained(output_dir)
            trainer.renderer.processor.save_pretrained(output_dir)

    DistributedInterface().barrier()
    logger.info_rank0(f"Model saved to {output_dir}")
