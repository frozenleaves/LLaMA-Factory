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

"""Expert Parallelism utilities for FSDP2+EP plugin."""

from .dispatcher import get_experts_forward_fn
from .dist_ops import all_to_all, gather_along_first_dim
from .parallel_state import ExpertParallelState


__all__ = [
    "ExpertParallelState",
    "all_to_all",
    "gather_along_first_dim",
    "get_experts_forward_fn",
]
