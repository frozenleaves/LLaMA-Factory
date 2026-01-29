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

"""The definition of trainer.

Init Phase:

1. Init batch generator.
2. Init optimizer (deepspeed).
3. Shard model.
4. Init optimizer (fsdp).
5. Init lr scheduler.

Train Phase:
1. Train Loop

"""

from abc import abstractmethod
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from ..accelerator.helper import ReduceOp
from ..accelerator.interface import Dim, DistributedInterface
from ..config import TrainingArguments
from ..plugins.trainer_plugins.distributed.hub import DistributedPlugin
from ..utils import logging
from ..utils.helper import compute_valid_tokens
from ..utils.types import BatchInput, HFModel, ModelOutput, Tensor, TorchDataset
from .utils.batching import BatchGenerator
from .utils.rendering import Renderer

logger = logging.get_logger(__name__)


class BaseTrainer:
    def __init__(
        self,
        args: TrainingArguments,
        model: HFModel,
        renderer: Renderer,
        train_dataset: TorchDataset,
    ) -> None:
        self.args = args
        self.model = model
        self.renderer = renderer
        self.train_dataset = train_dataset

        # info
        self.global_step = 0

        # cached variables
        self.device = DistributedInterface().current_device
        self.dp_size = DistributedInterface().get_world_size(Dim.DP)
        self.model_input_names = self.renderer.processor.model_input_names

        self._create_batch_generator()
        # Calculate num_training_steps: max_steps takes priority if set
        if self.args.max_steps is not None and self.args.max_steps > 0:
            self.num_training_steps = self.args.max_steps
        else:
            self.num_training_steps = self.args.num_train_epochs * len(self.train_batch_generator)

        if self.args.enable_activation_checkpointing:
            self.model.gradient_checkpointing_enable({"use_reentrant": False})

        # Initialize plugin system
        self._initialize_plugin_system()

        # Determine if sharding requires optimizer first (e.g., DeepSpeed)
        requires_optimizer_first = self._should_init_optimizer_before_sharding()
        
        if requires_optimizer_first:
            self._init_optimizer()
            self._shard_model()
        else:
            self._shard_model()
            self._init_optimizer()

        self._init_lr_scheduler()

    def _initialize_plugin_system(self) -> None:
        """Initialize the distributed plugin system and cache plugin hooks."""
        self._plugin_funcs: dict[str, Optional[Callable]] = {}
        
        if self.args.dist_config is not None:
            # Create the plugin instance based on configuration
            from ..plugins.trainer_plugins.distributed.hub import DistributedPlugin
            self.dist_plugin = DistributedPlugin(self.args.dist_config.name)
            # Cache hooks immediately after plugin creation
            self._cache_plugin_funcs()
    
    def _should_init_optimizer_before_sharding(self) -> bool:
        """Determine if the distributed strategy requires optimizer to be initialized before model sharding."""
        if self.args.dist_config is not None:
            # DeepSpeed requires optimizer to be initialized before sharding for proper integration
            return self.args.dist_config.name == "deepspeed"
        return False

    def _create_batch_generator(self) -> None:
        self.train_batch_generator = BatchGenerator(
            dataset=self.train_dataset,
            renderer=self.renderer,
            micro_batch_size=self.args.micro_batch_size,
            global_batch_size=self.args.global_batch_size,
            cutoff_len=self.args.cutoff_len,
            batching_workers=self.args.batching_workers,
            batching_strategy=self.args.batching_strategy,
        )

    def _shard_model(self) -> None:
        if self.args.dist_config is None:
            if DistributedInterface().get_world_size(Dim.DP) > 1:
                from torch.nn.parallel import DistributedDataParallel as DDP

                logger.warning_rank0(
                    "dist_config is None but distributed training is enabled; falling back to DistributedDataParallel."
                )
                device_ids = None if self.device.type == "cpu" else [self.device.index]
                self.model = DDP(self.model, device_ids=device_ids)
        else:
            optimizer = getattr(self, "optimizer", None)
            
            # Prepare batch configuration for distributed training
            batch_config = self._prepare_batch_config()
            
            # Apply the plugin to shard the model
            self.model = self.dist_plugin(self.model, self.args.dist_config, optimizer=optimizer, batch_config=batch_config)
            
            if self._plugin_funcs.get("post_shard"):
                self._plugin_funcs["post_shard"](self)

    def _prepare_batch_config(self) -> dict:
        """Prepare the batch configuration for deepspeed distributed training."""
        return {
            "micro_batch_size": self.args.micro_batch_size,
            "global_batch_size": self.train_batch_generator.global_batch_size,
            "num_micro_batch": self.train_batch_generator.num_micro_batch,
            "dp_size": self.dp_size,
            "bf16": self.args.bf16,
            "max_grad_norm": self.args.max_grad_norm,
            "num_training_steps": self.num_training_steps,
            "warmup_ratio": self.args.warmup_ratio,
            "learning_rate": self.args.learning_rate,
        }

    def _cache_plugin_funcs(self) -> None:
        """Cache distributed plugin functions to avoid repeated lookups."""
        if self.dist_plugin is None:
            return

        for func_name in ["post_shard", "init_lr_scheduler", "backward", "opt_step", "save_model"]:
            try:
                self._plugin_funcs[func_name] = self.dist_plugin[func_name]
            except ValueError:
                self._plugin_funcs[func_name] = None

    def _init_optimizer(self) -> None:
        """Init optimizer."""
        if self.args.optim_config is None:
            _trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(_trainable_params, lr=self.args.learning_rate)
        else:
            from ..plugins.trainer_plugins.optimizer import OptimizerPlugin

            self.optimizer = OptimizerPlugin(self.args.optim_config.name)(self.model, self.args.optim_config)

    def _init_lr_scheduler(self) -> None:
        """Init lr scheduler."""
        init_lr_scheduler_fn = self._plugin_funcs.get("init_lr_scheduler")
        if init_lr_scheduler_fn:
            lr_scheduler = init_lr_scheduler_fn(self)
            if lr_scheduler is not None:
                self.lr_scheduler = lr_scheduler
                return

        if self.args.lr_scheduler_config is None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: 1.0)
        else:
            from ..plugins.trainer_plugins.lr_scheduler import LRSchedulerPlugin

            self.lr_scheduler = LRSchedulerPlugin(self.args.lr_scheduler_config.name)(
                self.optimizer, self.num_training_steps, self.args.lr_scheduler_config
            )

    def compute_log_probs(self, model: HFModel, batch: BatchInput) -> Tensor:
        """Compute log probs.

        log_probs: Tensor of shape (batch_size, seq_len - 1)
        """
        batch_size, _ = batch["labels"].shape
        model_inputs = {
            k: v.to(self.device, non_blocking=True) for k, v in batch.items() if k in self.model_input_names
        }
        labels = batch["labels"].to(self.device, non_blocking=True)
        outputs: ModelOutput = model(**model_inputs)
        logits = outputs.logits.float()
        shift_labels = labels[..., 1:].contiguous().view(-1)
        shift_logits = logits[..., :-1, :].contiguous().view(shift_labels.size(0), -1)
        return -F.cross_entropy(shift_logits, shift_labels, reduction="none").view(batch_size, -1)

    @abstractmethod
    def compute_loss(self, batch: BatchInput) -> Tensor:
        """Compute the scalar loss."""
        ...

    def _backward(self, loss: Tensor) -> None:
        backward_fn = self._plugin_funcs.get("backward")
        if backward_fn:
            backward_fn(self, loss)
        else:
            loss.backward()

    def _opt_step(self) -> float:
        opt_step_fn = self._plugin_funcs.get("opt_step")
        if opt_step_fn:
            return opt_step_fn(self)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        if torch.is_tensor(grad_norm):
            grad_norm = grad_norm.item()

        # isfinite(): argument 'input' (position 1) must be Tensor, not float
        if not torch.isfinite(torch.tensor(grad_norm)):  # type: ignore # pyright: ignore [reportUnknownReturnType]
            logger.warning_rank0(f"Gradient norm is not finite: {grad_norm}")
        else:
            self.optimizer.step()

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return grad_norm

    def fit(self) -> None:
        """Train the model."""
        self.model.train()
        for epoch in range(self.args.num_train_epochs):
            self.train_batch_generator.set_epoch(epoch)
            for micro_batches in self.train_batch_generator:
                self.global_step += 1
                step_loss = 0
                step_valid_tokens = compute_valid_tokens(micro_batches)
                step_valid_tokens = DistributedInterface().all_reduce(step_valid_tokens, op=ReduceOp.SUM)
                for micro_batch in micro_batches:
                    loss = self.compute_loss(micro_batch)
                    mini_step_valid_tokens = compute_valid_tokens([micro_batch])
                    # fsdp uses mean reduction so we need to scale the loss by dp_size
                    loss = loss * mini_step_valid_tokens * self.dp_size / (step_valid_tokens + 1e-6)

                    self._backward(loss)
                    step_loss += loss.item()

                grad_norm = self._opt_step()

                step_loss, grad_norm = DistributedInterface().all_reduce([step_loss, grad_norm])
                DistributedInterface().sync()
                if DistributedInterface().get_rank() == 0:
                    print(f"Epoch {epoch}, Step {self.global_step}, Loss: {step_loss:.4f}, Grad Norm: {grad_norm:.4f}")

                # Check if max_steps is reached
                if self.global_step >= self.num_training_steps:
                    logger.info_rank0(f"Reached max_steps ({self.num_training_steps}), stopping training.")
                    return

    def save_model(self) -> None:
        """Save the model."""
        save_model_fn = self._plugin_funcs.get("save_model")
        if save_model_fn:
            return save_model_fn(self)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.args.output_dir)
        self.renderer.processor.save_pretrained(self.args.output_dir)
        logger.info_rank0(f"Model saved to {self.args.output_dir}")
