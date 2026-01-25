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
def shard_model_fsdp2(model: HFModel, dist_config: PluginConfig) -> HFModel:
    from .fsdp2 import FSDP2Engine

    return FSDP2Engine(dist_config).shard_model(model)


@DistributedPlugin("deepspeed").register()
def shard_model_deepspeed(model: HFModel, dist_config: PluginConfig) -> HFModel:
    return model


@DistributedPlugin("fsdp2_ep").register()
def shard_model_fsdp2_ep(model: HFModel, dist_config: PluginConfig) -> HFModel:
    """Shard model with FSDP2 + Expert Parallelism.

    This plugin supports MoE models with expert parallelism:
    - EP (Expert Parallelism): Distributes experts across ranks
    - EFSDP (Expert FSDP): Additional FSDP sharding for expert parameters
    - Standard FSDP: For non-expert model parameters

    Configuration options in dist_config:
        - ep_size: Expert parallel size (default: 1)
        - efsdp_size: Expert FSDP size (default: 1)
        - mixed_precision: "bf16", "fp16", or "fp32"
        - expert_modules: List of patterns to match expert modules (e.g., ["*.mlp.experts"])
        - reshard_after_forward: Whether to reshard after forward pass
        - offload_params: Whether to offload parameters to CPU
        - dcp_path: Path for distributed checkpoint
    """
    from .fsdp2_ep import FSDP2EPEngine

    return FSDP2EPEngine(dist_config).shard_model(model)
