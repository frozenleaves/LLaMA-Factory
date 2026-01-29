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

from ....config.arg_utils import PluginConfig
from ....utils.plugin import BasePlugin
from ....utils.types import HFModel


class DistributedPlugin(BasePlugin):
    def __call__(self, model: HFModel, dist_config: PluginConfig, **kwargs) -> HFModel:
        return super().__call__(model, dist_config, **kwargs)


@DistributedPlugin("fsdp2").register()
def shard_model_fsdp2(model: HFModel, dist_config: PluginConfig, **kwargs) -> HFModel:
    from .fsdp2 import FSDP2Engine

    return FSDP2Engine(dist_config).shard_model(model)


@DistributedPlugin("deepspeed").register()
def shard_model_deepspeed(model: HFModel, dist_config: PluginConfig, **kwargs) -> HFModel:
    from .deepspeed import DeepSpeedEngine

    return DeepSpeedEngine(dist_config, batch_config=kwargs.get("batch_config")).shard_model(
        model, optimizer=kwargs.get("optimizer")
    )


@DistributedPlugin("deepspeed").register("post_shard")
def post_shard_deepspeed(trainer) -> None:
    from .deepspeed import post_shard

    return post_shard(trainer)


@DistributedPlugin("deepspeed").register("init_lr_scheduler")
def init_lr_scheduler_deepspeed(trainer):
    from .deepspeed import init_lr_scheduler

    return init_lr_scheduler(trainer)


@DistributedPlugin("deepspeed").register("backward")
def backward_deepspeed(trainer, loss) -> None:
    from .deepspeed import backward

    return backward(trainer, loss)


@DistributedPlugin("deepspeed").register("opt_step")
def opt_step_deepspeed(trainer) -> float:
    from .deepspeed import opt_step

    return opt_step(trainer)


@DistributedPlugin("deepspeed").register("save_model")
def save_model_deepspeed(trainer) -> None:
    from .deepspeed import save_model

    return save_model(trainer)
