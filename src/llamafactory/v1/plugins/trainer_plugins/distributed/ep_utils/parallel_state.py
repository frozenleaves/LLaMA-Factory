# Copyright 2025 the LlamaFactory team.
#
# This code is inspired by the MindSpeed library.
# https://gitee.com/ascend/MindSpeed/blob/master/mindspeed/lite/distributed/parallel_state.py
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

"""Expert Parallel State management for FSDP2+EP plugin."""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from .....accelerator.helper import get_current_accelerator, get_process_group_backend
from .....utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ExpertParallelState:
    """Manages device mesh for Expert Parallelism.

    This creates an independent device mesh for expert parallel dimensions:
    - edp (Expert Data Parallel): Data parallelism within expert groups
    - efsdp (Expert Fully Shard): FSDP for expert parameters
    - ep (Expert Parallel): Expert parallelism dimension

    Args:
        ep_size: Expert parallel size. Number of ranks to distribute experts across.
        efsdp_size: Expert FSDP size. Number of ranks for FSDP within each EP group.
    """

    ep_size: int = 1
    efsdp_size: int = 1
    device_mesh: Optional[DeviceMesh] = field(default=None, init=False)
    _edp_size: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group must be initialized before creating ExpertParallelState.")

        world_size = dist.get_world_size()

        # Validate configuration
        if world_size % (self.ep_size * self.efsdp_size) != 0:
            raise ValueError(
                f"World size ({world_size}) must be divisible by ep_size * efsdp_size "
                f"({self.ep_size} * {self.efsdp_size} = {self.ep_size * self.efsdp_size})."
            )

        # Calculate expert data parallel size
        self._edp_size = world_size // (self.ep_size * self.efsdp_size)

        # Create device mesh for expert parallelism: (edp, efsdp, ep)
        mesh_dim_names = ("edp", "efsdp", "ep")
        mesh_shape = (self._edp_size, self.efsdp_size, self.ep_size)

        device = get_current_accelerator()
        device_type = device.type

        self.device_mesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )

        rank = dist.get_rank()
        if rank == 0:
            logger.info(
                f"ExpertParallelState initialized: "
                f"ep_size={self.ep_size}, efsdp_size={self.efsdp_size}, edp_size={self._edp_size}, "
                f"device_mesh={self.device_mesh}"
            )

    @property
    def edp_size(self) -> int:
        """Expert data parallel size."""
        return self._edp_size

    @property
    def ep_mesh(self) -> DeviceMesh:
        """Device mesh for expert parallel dimension."""
        return self.device_mesh["ep"]

    @property
    def efsdp_mesh(self) -> DeviceMesh:
        """Device mesh for expert FSDP dimension."""
        return self.device_mesh["efsdp"]

    @property
    def edp_mesh(self) -> DeviceMesh:
        """Device mesh for expert data parallel dimension."""
        return self.device_mesh["edp"]

    @property
    def ep_group(self) -> dist.ProcessGroup:
        """Process group for expert parallel communication."""
        return self.ep_mesh.get_group()

    @property
    def efsdp_group(self) -> dist.ProcessGroup:
        """Process group for expert FSDP communication."""
        return self.efsdp_mesh.get_group()

    @property
    def edp_group(self) -> dist.ProcessGroup:
        """Process group for expert data parallel communication."""
        return self.edp_mesh.get_group()

    @property
    def ep_rank(self) -> int:
        """Rank within expert parallel group."""
        return self.ep_mesh.get_local_rank()

    @property
    def efsdp_rank(self) -> int:
        """Rank within expert FSDP group."""
        return self.efsdp_mesh.get_local_rank()

    @property
    def edp_rank(self) -> int:
        """Rank within expert data parallel group."""
        return self.edp_mesh.get_local_rank()

    @property
    def gradient_divide_factor(self) -> float:
        """Factor to divide gradients by for proper averaging across all EP dimensions."""
        return float(self.ep_size * self.efsdp_size * self._edp_size)

    def is_ep_enabled(self) -> bool:
        """Check if expert parallelism is enabled."""
        return self.ep_size > 1

    def is_efsdp_enabled(self) -> bool:
        """Check if expert FSDP is enabled."""
        return self.efsdp_size > 1
