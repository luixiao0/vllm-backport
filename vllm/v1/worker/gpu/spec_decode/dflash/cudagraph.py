# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from collections.abc import Callable, Mapping

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import maybe_prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionState,
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup


def _prepare_dflash_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    skip_attn: bool,
    causal: bool | Mapping[int, bool],
) -> AttentionState:
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    attn_metadata = None
    if not skip_attn:
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        input_batch.dcp_local_seq_lens = maybe_prepare_dcp_local_seq_lens(
            input_buffers.dcp_local_seq_lens,
            input_batch.seq_lens,
            input_batch.num_reqs,
            block_tables.cp_size,
            block_tables.cp_rank,
            block_tables.cp_interleave,
            num_reqs_padded=input_batch.num_reqs_after_padding,
        )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=num_tokens // num_reqs,
            seq_lens=input_batch.seq_lens,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            max_seq_len=max_model_len,
            block_tables=input_block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            for_cudagraph_capture=True,
            causal=causal,
        )
    return AttentionState(attn_metadata, slot_mappings_by_layer)


class DFlashCudaGraphManager(CudaGraphManager):
    """DFlash CudaGraphManager for the parallel-drafting query forward,
    building its own attention metadata from scratch.

    ``capture_sizes`` overrides the derived draft-token buckets; an empty list
    disables capture. Overrides must contain whole query blocks.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
        *,
        capture_sizes: list[int] | None = None,
    ):
        graph_config = copy.copy(vllm_config)
        graph_config.compilation_config = copy.copy(vllm_config.compilation_config)
        if capture_sizes is None:
            capture_sizes = _draft_capture_sizes(vllm_config, decode_query_len)
        else:
            capture_sizes = list(capture_sizes)
            if any(size <= 0 or size % decode_query_len for size in capture_sizes):
                raise ValueError("Draft capture sizes must be positive query blocks")
            max_tokens = min(
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.scheduler_config.max_num_seqs * decode_query_len,
            )
            if any(size > max_tokens for size in capture_sizes):
                raise ValueError(
                    f"Draft capture exceeds draft buffer capacity ({max_tokens} tokens)"
                )
        graph_config.compilation_config.cudagraph_capture_sizes = capture_sizes
        graph_config.compilation_config.max_cudagraph_capture_size = max(
            capture_sizes, default=0
        )
        # A parallel draft always runs its fixed query block, independently of
        # the target's dynamic verification budget.
        graph_config.speculative_config = None
        super().__init__(
            graph_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
            varlen_decode=varlen_decode,
        )

    def capture(
        self,
        forward_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        causal: bool | Mapping[int, bool],
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            attn_state = _prepare_dflash_inputs_to_capture(
                num_reqs,
                num_tokens,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                max_model_len,
                skip_attn=(desc.cg_mode == CUDAGraphMode.PIECEWISE),
                causal=causal,
            )
            attn_metadata, slot_mappings = attn_state

            return lambda cg_mode: forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
            )

        super().capture(create_forward_fn, progress_bar_desc)


def _draft_capture_sizes(vllm_config: VllmConfig, query_len: int) -> list[int]:
    compilation_config = vllm_config.compilation_config
    target_sizes = compilation_config.cudagraph_capture_sizes
    if not target_sizes:
        return []
    target_query_len = vllm_config.num_speculative_tokens + 1
    target_cap = compilation_config.max_cudagraph_capture_size
    max_reqs = min(
        vllm_config.scheduler_config.max_num_seqs,
        vllm_config.scheduler_config.max_num_batched_tokens // query_len,
        target_cap // query_len,
    )
    # Keep existing legal draft buckets and translate the target's request
    # buckets. Small batches get exact shapes; large grids remain sparse.
    req_counts = {
        round_up(size, width) // width
        for size in target_sizes
        for width in (query_len, target_query_len)
    }
    req_counts.update(range(1, min(8, max_reqs) + 1))
    return sorted(n * query_len for n in req_counts if 0 < n <= max_reqs)
