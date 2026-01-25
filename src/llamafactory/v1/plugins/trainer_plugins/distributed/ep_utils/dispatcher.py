# Copyright 2025 the LlamaFactory team.
#
# This code is inspired by the MindSpeed library.
# https://gitee.com/ascend/MindSpeed/blob/master/mindspeed/lite/distributed/expert_parallel/dispatcher.py
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

"""Token dispatcher for Expert Parallelism.

This module implements the token routing logic for MoE models with expert parallelism:
1. Dispatch: Route tokens to appropriate experts across EP ranks via AllToAll
2. Compute: Execute expert computation on local tokens
3. Combine: Gather results back via AllToAll and apply routing weights
"""

from collections.abc import Callable
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from .....accelerator.helper import get_current_accelerator
from .dist_ops import all_to_all, gather_along_first_dim


def permute(
    tokens: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute tokens according to expert indices.

    Args:
        tokens: Input tokens of shape (batch_size * seq_len, hidden_dim).
        indices: Expert indices of shape (batch_size * seq_len,) or (batch_size * seq_len, top_k).

    Returns:
        Tuple of (permuted_tokens, sorted_indices) for later unpermutation.
    """
    topk = 1 if indices.dim() == 1 else indices.size(1)
    indices_dtype = indices.dtype
    # Sort indices to group tokens by expert
    sorted_indices = torch.argsort(indices.float().view(-1), stable=True).to(indices_dtype)
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unpermute tokens back to original order and apply routing weights.

    Args:
        permuted_tokens: Tokens after expert computation.
        sorted_indices: Indices from permute operation.
        probs: Routing weights of shape (batch_size * seq_len, top_k). If None, no weighting is applied.

    Returns:
        Unpermuted tokens with routing weights applied.
    """
    if permuted_tokens.size(0) != sorted_indices.numel():
        raise AssertionError(
            f"permuted tokens size ({permuted_tokens.size(0)}) != sorted indices size ({sorted_indices.numel()})"
        )

    num_tokens, topk = (permuted_tokens.size(0), 1) if probs is None else (probs.numel(), probs.size(1))
    unpermuted_tokens = torch.zeros(
        [num_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))

    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)

    return unpermuted_tokens.sum(dim=1)


def dispatch_preprocess(
    ep_group: dist.ProcessGroup,
    top_k_index: torch.Tensor,
    num_global_experts: int,
    expert_ids_per_ep_rank: Optional[torch.Tensor] = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Preprocess for dispatch: compute token counts and split sizes.

    Args:
        ep_group: Expert parallel process group.
        top_k_index: Expert indices of shape (batch_size * seq_len, top_k).
        num_global_experts: Total number of experts across all EP ranks.
        expert_ids_per_ep_rank: Mapping from global expert ID to local expert ID.

    Returns:
        Tuple of ((num_tokens_per_local_expert, global_indices), (input_splits, output_splits)).
    """
    ep_size = dist.get_world_size(ep_group)
    ep_rank = dist.get_rank(ep_group)
    num_local_experts = num_global_experts // ep_size
    local_experts_start_id = num_local_experts * ep_rank
    local_experts_end_id = local_experts_start_id + num_local_experts

    # Count tokens per expert: [B*S, K] --> [E]
    num_local_tokens_per_expert = torch.bincount(top_k_index.view(-1), minlength=num_global_experts)

    # Gather counts from all EP ranks: [E] --> [EP*E]
    num_global_tokens_per_expert, _ = gather_along_first_dim(num_local_tokens_per_expert, ep_group)

    # Get counts for local experts from each EP rank: [EP*E] --> [EP, local_E]
    num_global_tokens_per_local_expert = num_global_tokens_per_expert.reshape(ep_size, num_global_experts)[
        :, local_experts_start_id:local_experts_end_id
    ]

    # Total tokens per local expert: [EP, local_E] --> [local_E]
    num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)

    # Input splits: tokens from this rank to each EP rank
    # [E] --> [EP, local_E] --> [EP]
    input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).cpu()

    # Output splits: tokens from each EP rank to this rank
    # [EP, local_E] --> [EP]
    output_splits = num_global_tokens_per_local_expert.sum(dim=-1).cpu()

    # Global indices for unpermutation after receiving tokens
    if expert_ids_per_ep_rank is not None:
        global_indices = torch.repeat_interleave(expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel())
    else:
        # Create default mapping
        device = get_current_accelerator()
        expert_ids = torch.tensor(
            [i % num_local_experts for i in range(num_global_experts)],
            dtype=torch.int32,
            device=device,
        )
        global_indices = torch.repeat_interleave(expert_ids, num_global_tokens_per_local_expert.ravel())

    return (num_tokens_per_local_expert, global_indices), (input_splits, output_splits)


def alltoall_dispatch(
    ep_group: dist.ProcessGroup,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    permute_indices: tuple[torch.Tensor, torch.Tensor],
    split_sizes: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Dispatch tokens to experts via AllToAll communication.

    Args:
        ep_group: Expert parallel process group.
        hidden_states: Input hidden states.
        top_k_index: Expert indices for each token.
        permute_indices: Tuple of (num_tokens_per_local_expert, global_indices).
        split_sizes: Tuple of (input_splits, output_splits).

    Returns:
        Tuple of (dispatched_hidden_states, unpermute_indices).
    """
    local_indices, global_indices = permute_indices
    input_splits, output_splits = split_sizes

    # Permute tokens by expert assignment
    hidden_states, unpermute_indices1 = permute(hidden_states, top_k_index)

    # AllToAll: send tokens to appropriate EP ranks
    hidden_states = all_to_all(ep_group, hidden_states, output_splits.tolist(), input_splits.tolist())

    # Permute received tokens by local expert assignment
    hidden_states, unpermute_indices2 = permute(hidden_states, global_indices)

    return hidden_states, (unpermute_indices1, unpermute_indices2)


def alltoall_combine(
    ep_group: dist.ProcessGroup,
    hidden_states: torch.Tensor,
    top_k_weights: torch.Tensor,
    unpermute_indices: tuple[torch.Tensor, torch.Tensor],
    split_sizes: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Combine expert outputs via AllToAll communication.

    Args:
        ep_group: Expert parallel process group.
        hidden_states: Expert output hidden states.
        top_k_weights: Routing weights for combining expert outputs.
        unpermute_indices: Indices for unpermutation from dispatch phase.
        split_sizes: Tuple of (input_splits, output_splits).

    Returns:
        Combined hidden states after AllToAll and weighted sum.
    """
    unpermute_indices1, unpermute_indices2 = unpermute_indices
    input_splits, output_splits = split_sizes

    # Unpermute local expert outputs
    hidden_states = unpermute(hidden_states, unpermute_indices2)

    # AllToAll: gather outputs back to original ranks
    hidden_states = all_to_all(ep_group, hidden_states, input_splits.tolist(), output_splits.tolist())

    # Unpermute and apply routing weights
    hidden_states = unpermute(hidden_states, unpermute_indices1, top_k_weights)

    return hidden_states


def _ensure_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is a local tensor, not DTensor.

    Args:
        tensor: Input tensor (may be DTensor).

    Returns:
        Local tensor.
    """
    if hasattr(tensor, "to_local"):
        return tensor.to_local()
    return tensor


def eager_experts_computation(
    hidden_states: torch.Tensor,
    split_list: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    act_fn: Callable,
) -> torch.Tensor:
    """Compute expert MLP outputs using eager (non-fused) operations.

    Args:
        hidden_states: Input hidden states of shape (total_tokens, hidden_dim).
        split_list: Number of tokens for each local expert.
        gate_up_weights: Gate and up projection weights.
        down_weights: Down projection weights.
        act_fn: Activation function.

    Returns:
        Expert output hidden states.
    """
    # Ensure weights are local tensors (not DTensor)
    gate_up_weights = _ensure_local_tensor(gate_up_weights)
    down_weights = _ensure_local_tensor(down_weights)
    hidden_states = _ensure_local_tensor(hidden_states)

    # Split hidden states by expert
    splits = split_list.tolist()
    hidden_splits = torch.split(hidden_states, splits, dim=0)

    outputs = []
    for i, (h, n_tokens) in enumerate(zip(hidden_splits, splits)):
        if n_tokens == 0:
            continue

        # Get weights for this expert (ensure they are local tensors)
        gate_up_w = gate_up_weights[i]
        down_w = down_weights[i]

        # Ensure indexed weights are also local tensors
        gate_up_w = _ensure_local_tensor(gate_up_w)
        down_w = _ensure_local_tensor(down_w)

        # Gate-up projection
        gate_up = torch.matmul(h, gate_up_w.T)
        gate, up = gate_up.chunk(2, dim=-1)
        # Activation and down projection
        act = act_fn(gate) * up
        out = torch.matmul(act, down_w.T)
        outputs.append(out)

    if outputs:
        return torch.cat(outputs, dim=0)
    else:
        return hidden_states.new_empty(0, down_weights.shape[-1])


def dispatch_mlp_combine(
    ep_group: dist.ProcessGroup,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor],
    act_fn: Callable,
    num_global_experts: int,
    expert_ids_per_ep_rank: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full dispatch-compute-combine pipeline for expert MLP.

    Args:
        ep_group: Expert parallel process group.
        hidden_states: Input hidden states.
        top_k_index: Expert indices for each token.
        top_k_weights: Routing weights for each token.
        weights: Tuple of (gate_up_weights, down_weights).
        act_fn: Activation function.
        num_global_experts: Total number of experts.
        expert_ids_per_ep_rank: Mapping from global to local expert IDs.

    Returns:
        Output hidden states after expert computation.
    """
    gate_up_weights, down_weights = weights

    # Preprocess: compute token counts and split sizes
    permute_indices, split_sizes = dispatch_preprocess(
        ep_group, top_k_index, num_global_experts, expert_ids_per_ep_rank
    )

    # Dispatch tokens to experts via AllToAll
    hidden_states, unpermute_indices = alltoall_dispatch(
        ep_group, hidden_states, top_k_index, permute_indices, split_sizes
    )

    # Expert computation
    hidden_states = eager_experts_computation(hidden_states, permute_indices[0], gate_up_weights, down_weights, act_fn)

    # Combine outputs via AllToAll
    hidden_states = alltoall_combine(ep_group, hidden_states, top_k_weights, unpermute_indices, split_sizes)

    return hidden_states


def get_experts_forward_fn(
    ep_group: dist.ProcessGroup,
    num_global_experts: int,
    num_local_experts: int,
    ep_rank: int,
) -> Callable:
    """Create a forward function for expert module with EP communication.

    Args:
        ep_group: Expert parallel process group.
        num_global_experts: Total number of experts.
        num_local_experts: Number of experts on this rank.
        ep_rank: Rank in the expert parallel group.

    Returns:
        A forward function that can replace the original expert forward.
    """
    # Pre-compute expert ID mapping
    device = get_current_accelerator()
    expert_ids_per_ep_rank = torch.tensor(
        [i % num_local_experts for i in range(num_global_experts)],
        dtype=torch.int32,
        device=device,
    )

    def experts_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function with expert parallelism.

        Args:
            self: The expert module.
            hidden_states: Input hidden states of shape (batch_size * seq_len, hidden_dim).
            top_k_index: Expert indices of shape (batch_size * seq_len, top_k).
            top_k_weights: Routing weights of shape (batch_size * seq_len, top_k).

        Returns:
            Output hidden states.
        """
        hidden_states_shape = hidden_states.shape

        # Get weights - handle different expert module structures
        # Try common attribute names for expert weights
        gate_up_proj = None
        down_proj = None

        # Check for gate_up_proj (combined gate and up projection)
        if hasattr(self, "gate_up_proj"):
            gate_up_proj = self.gate_up_proj
            if hasattr(gate_up_proj, "weight"):
                gate_up_proj = gate_up_proj.weight
        elif hasattr(self, "w1") and hasattr(self, "w3"):
            # Some models use w1 (gate) and w3 (up) separately
            # We'd need to handle this differently - for now skip
            pass

        # Check for down_proj
        if hasattr(self, "down_proj"):
            down_proj = self.down_proj
            if hasattr(down_proj, "weight"):
                down_proj = down_proj.weight
        elif hasattr(self, "w2"):
            down_proj = self.w2
            if hasattr(down_proj, "weight"):
                down_proj = down_proj.weight

        if gate_up_proj is None or down_proj is None:
            raise RuntimeError(
                f"Cannot find expert weights in module {type(self).__name__}. "
                "Expected 'gate_up_proj' and 'down_proj' attributes."
            )

        # Convert DTensor to local tensor if needed
        if hasattr(gate_up_proj, "to_local"):
            gate_up_proj = gate_up_proj.to_local()
        if hasattr(down_proj, "to_local"):
            down_proj = down_proj.to_local()

        # Ensure hidden_states is also a local tensor
        if hasattr(hidden_states, "to_local"):
            hidden_states = hidden_states.to_local()

        weights = (gate_up_proj, down_proj)
        act_fn = self.act_fn if hasattr(self, "act_fn") else torch.nn.functional.silu

        hidden_states = dispatch_mlp_combine(
            ep_group,
            hidden_states.view(-1, hidden_states.size(-1)),
            top_k_index,
            top_k_weights,
            weights,
            act_fn,
            num_global_experts,
            expert_ids_per_ep_rank,
        )

        return hidden_states.view(*hidden_states_shape)

    return experts_forward
