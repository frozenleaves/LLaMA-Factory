# Copyright 2025 the LlamaFactory team.
#
# This code is inspired by the MindSpeed library.
# https://gitee.com/ascend/MindSpeed/blob/master/mindspeed/lite/distributed/dist_ops.py
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

"""Distributed operations for Expert Parallelism."""

from typing import Optional, Sequence

import torch
import torch.distributed as dist

from .....accelerator.helper import get_current_accelerator


class _AllToAll(torch.autograd.Function):
    """AllToAll autograd function with support for unequal splits."""

    @staticmethod
    def forward(
        ctx,
        group: dist.ProcessGroup,
        inputs: torch.Tensor,
        output_split_sizes: Optional[Sequence[int]],
        input_split_sizes: Optional[Sequence[int]],
    ) -> torch.Tensor:
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return inputs

        inputs = inputs.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(inputs)
        else:
            # Unequal split (all2all-v)
            device = get_current_accelerator()
            output = inputs.new_empty(
                size=[sum(output_split_sizes)] + list(inputs.size()[1:]),
                dtype=inputs.dtype,
                device=device,
            )

        # Convert to list if numpy array
        out_splits = list(output_split_sizes) if output_split_sizes is not None else None
        in_splits = list(input_split_sizes) if input_split_sizes is not None else None

        dist.all_to_all_single(
            output,
            inputs,
            output_split_sizes=out_splits,
            input_split_sizes=in_splits,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward function - reverse the AllToAll operation."""
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )


def all_to_all(
    group: dist.ProcessGroup,
    inputs: torch.Tensor,
    output_split_sizes: Optional[Sequence[int]] = None,
    input_split_sizes: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Wrapper for AllToAll autograd function.

    Args:
        group: The process group for communication.
        inputs: Input tensor to be distributed.
        output_split_sizes: Sizes of output splits for each rank (for unequal splits).
        input_split_sizes: Sizes of input splits for each rank (for unequal splits).

    Returns:
        Output tensor after AllToAll communication.
    """
    return _AllToAll.apply(group, inputs, output_split_sizes, input_split_sizes)


def gather_along_first_dim(
    input_tensor: torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
) -> tuple[torch.Tensor, Optional[dist.Work]]:
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor: Input tensor to gather.
        group: The process group for communication.
        async_op: Whether to perform the operation asynchronously.

    Returns:
        A tuple of (output tensor, async handle if async_op else None).
    """
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return input_tensor, None

    dim_size = list(input_tensor.size())
    dim_size[0] = dim_size[0] * world_size

    device = get_current_accelerator()
    output = torch.empty(dim_size, dtype=input_tensor.dtype, device=device)
    handle = dist.all_gather_into_tensor(
        output,
        input_tensor.contiguous(),
        group=group,
        async_op=async_op,
    )

    return output, handle
