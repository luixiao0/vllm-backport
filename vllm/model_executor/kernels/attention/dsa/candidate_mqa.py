# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-only FP8 paged MQA for SM80, with the native top-k contract.

Unlike the upstream SM100 sparse-MQA path this retains FP8 cache bytes, FP32
head weights/logits, and the native token ordering. Prefill remains dense.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.mqa_logits_triton import _get_e4m3fn_bf16_lut


@triton.jit
def _prepare_candidates(
    candidates,
    lengths,
    blocks,
    keep,
    compact_lengths,
    q_bytes,
    q_bf16,
    lut,
    q_stride_batch,
    q_stride_token,
    q_stride_head,
    q_stride_dim,
    candidate_stride_row,
    candidate_stride_col,
    length_stride_batch,
    length_stride_token,
    NEXT_N: tl.constexpr,
    CANDIDATES: tl.constexpr,
    CAPACITY: tl.constexpr,
    CANDIDATE_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    MAX_LEN: tl.constexpr,
    LENGTHS_2D: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offset = tl.arange(0, TILE)
    for half in tl.static_range(2):
        query_offset = offset + half * TILE
        byte = tl.load(
            q_bytes
            + row // NEXT_N * q_stride_batch
            + row % NEXT_N * q_stride_token
            + query_offset // 128 * q_stride_head
            + query_offset % 128 * q_stride_dim
        )
        query = tl.load(lut + byte.to(tl.uint32))
        tl.store(q_bf16 + row * 4096 + query_offset, query)
    length = tl.load(
        lengths
        + row // NEXT_N * length_stride_batch
        + (row % NEXT_N if LENGTHS_2D else 0) * length_stride_token
    )
    if not LENGTHS_2D:
        length = length - NEXT_N + row % NEXT_N + 1
    length = tl.minimum(tl.maximum(length, 0), MAX_LEN)
    candidate = tl.load(
        candidates + row * candidate_stride_row + offset * candidate_stride_col,
        offset < CANDIDATES,
        other=-1,
    )
    filler_blocks: tl.constexpr = (TOP_K + CANDIDATE_SIZE - 1) // CANDIDATE_SIZE
    short = length <= CANDIDATES * CANDIDATE_SIZE
    initial = tl.where(short | (offset < filler_blocks), offset, -1)
    tl.store(blocks + row * CAPACITY + offset, initial, offset < CAPACITY)
    tl.store(keep + row * CAPACITY + offset, 0, offset < CAPACITY)
    tail = offset + TILE
    tl.store(
        blocks + row * CAPACITY + tail,
        tl.where(short, tail, -1),
        tail < CAPACITY,
    )
    tl.store(keep + row * CAPACITY + tail, 0, tail < CAPACITY)
    tl.debug_barrier()
    valid = (candidate >= 0) & (candidate < tl.cdiv(length, CANDIDATE_SIZE))
    # Prefix fillers preserve native lowest-index -inf ties. Short rows use
    # identity blocks, so marking membership avoids sorting altogether.
    tl.atomic_or(
        keep + row * CAPACITY + candidate,
        1,
        valid & (short | (candidate < filler_blocks)),
        sem="relaxed",
    )
    if length > CANDIDATES * CANDIDATE_SIZE:
        # Sort candidates after the fixed prefix, halving the sorting network
        # compared with padding candidates and fillers to 4096 lanes.
        key = tl.where(valid & (candidate >= filler_blocks), candidate, 0x7FFFFFFF)
        key = tl.sort(key, descending=False)
        previous = tl.gather(key, tl.maximum(offset - 1, 0), 0)
        unique = (key != 0x7FFFFFFF) & ((offset == 0) | (key != previous))
        destination = filler_blocks + tl.cumsum(unique.to(tl.int32)) - 1
        tl.store(blocks + row * CAPACITY + destination, key, unique)
        tl.store(keep + row * CAPACITY + destination, 1, unique)
        compact_length = (filler_blocks + tl.sum(unique.to(tl.int32))) * CANDIDATE_SIZE
    else:
        compact_length = length
    tl.store(compact_lengths + row, compact_length)


@triton.jit
def _candidate_paged_mqa(
    q,
    cache,
    weights,
    lut,
    lengths,
    table,
    blocks,
    keep,
    logits,
    q_stride_batch,
    q_stride_token,
    q_stride_head,
    q_stride_dim,
    cache_stride,
    weight_stride_row,
    weight_stride_head,
    length_stride_batch,
    length_stride_token,
    table_stride_batch,
    table_stride_block,
    NEXT_N: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CANDIDATE_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    TOP_K: tl.constexpr,
    MAX_LEN: tl.constexpr,
    LENGTHS_2D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // NEXT_N
    token = row % NEXT_N
    position = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    block = tl.load(
        blocks + row * CAPACITY + position // CANDIDATE_SIZE,
        position < CAPACITY * CANDIDATE_SIZE,
        other=-1,
    )
    selected = tl.load(
        keep + row * CAPACITY + position // CANDIDATE_SIZE,
        position < CAPACITY * CANDIDATE_SIZE,
        other=0,
    )
    length = tl.load(
        lengths
        + batch * length_stride_batch
        + (token if LENGTHS_2D else 0) * length_stride_token
    )
    if not LENGTHS_2D:
        length = length - NEXT_N + token + 1
    length = tl.minimum(tl.maximum(length, 0), MAX_LEN)
    context_position = block * CANDIDATE_SIZE + position % CANDIDATE_SIZE
    valid = (block >= 0) & (context_position < length) & (selected != 0)
    output = tl.full((BLOCK_N,), -float("inf"), tl.float32)
    if (length > TOP_K) & (tl.sum(valid.to(tl.int32), 0) > 0):
        head = tl.arange(0, BLOCK_H)
        dim = tl.arange(0, DIM)
        query = tl.load(
            q
            + batch * q_stride_batch
            + token * q_stride_token
            + head[:, None] * q_stride_head
            + dim[None, :] * q_stride_dim,
            head[:, None] < HEADS,
            other=0,
        )
        physical = tl.load(
            table
            + batch * table_stride_batch
            + context_position // PAGE_SIZE * table_stride_block,
            valid,
            other=0,
        ).to(tl.int64)
        page_offset = context_position % PAGE_SIZE
        byte = tl.load(
            cache
            + physical[:, None] * cache_stride
            + page_offset[:, None] * DIM
            + dim[None, :],
            valid[:, None],
            other=0,
        )
        key = tl.load(lut + byte.to(tl.uint32))
        scale = tl.load(
            (cache + physical * cache_stride + PAGE_SIZE * DIM).to(
                tl.pointer_type(tl.float32)
            )
            + page_offset,
            valid,
            other=0,
        )
        score = tl.dot(query, tl.trans(key)) * scale[None, :]
        weight = tl.load(
            weights + row * weight_stride_row + head * weight_stride_head,
            head < HEADS,
            other=0,
        )
        score = tl.where(score > 0, score, 0.0) * weight[:, None]
        output = tl.where(valid, tl.sum(score, 0), -float("inf"))
    tl.store(
        logits + row * CAPACITY * CANDIDATE_SIZE + position,
        output,
        position < CAPACITY * CANDIDATE_SIZE,
    )


@triton.jit
def _restore_indices(
    compact_indices,
    logits,
    blocks,
    lengths,
    output,
    output_stride_row,
    output_stride_col,
    length_stride_batch,
    length_stride_token,
    NEXT_N: tl.constexpr,
    CAPACITY: tl.constexpr,
    CANDIDATE_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    MAX_LEN: tl.constexpr,
    LENGTHS_2D: tl.constexpr,
    DENSE_FILTERED: tl.constexpr,
    INPLACE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offset = tl.arange(0, TOP_K)
    length = tl.load(
        lengths
        + row // NEXT_N * length_stride_batch
        + (row % NEXT_N if LENGTHS_2D else 0) * length_stride_token
    )
    if not LENGTHS_2D:
        length = length - NEXT_N + row % NEXT_N + 1
    length = tl.minimum(tl.maximum(length, 0), MAX_LEN)
    identity_limit: tl.constexpr = (
        CAPACITY - (TOP_K + CANDIDATE_SIZE - 1) // CANDIDATE_SIZE
    ) * CANDIDATE_SIZE
    if INPLACE and length <= identity_limit:
        return
    if length <= TOP_K:
        tl.store(
            output + row * output_stride_row + offset * output_stride_col,
            tl.where(offset < length, offset, -1),
        )
    else:
        index = tl.load(compact_indices + row * TOP_K + offset)
        block = tl.load(blocks + row * CAPACITY + index // CANDIDATE_SIZE)
        restored = block * CANDIDATE_SIZE + index % CANDIDATE_SIZE
        destination = offset
        # The long-row persistent radix emits >pivot first, then ==pivot;
        # FilteredTopK (and the short path) emit ascending context indices.
        if not DENSE_FILTERED and length > 32768:
            score = tl.load(logits + row * CAPACITY * CANDIDATE_SIZE + index)
            bits = score.to(tl.uint32, bitcast=True)
            key = tl.where(bits & 0x80000000 != 0, bits ^ 0xFFFFFFFF, bits | 0x80000000)
            pivot = tl.min(key, 0)
            greater = key > pivot
            destination = tl.where(
                greater,
                tl.cumsum(greater.to(tl.int32)) - 1,
                tl.sum(greater.to(tl.int32)) + tl.cumsum((~greater).to(tl.int32)) - 1,
            )
        if INPLACE:
            tl.debug_barrier()
        tl.store(
            output + row * output_stride_row + destination * output_stride_col,
            restored,
        )


def fp8_candidate_paged_mqa_topk(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
    topk_indices: torch.Tensor,
    max_model_len: int,
    workspace: torch.Tensor,
) -> None:
    """Score only candidate positions and restore exact native decode ordering.

    Inputs use the ordinary FP8 paged-indexer layout. Candidate IDs are local
    blocks of compressed positions. Negative, repeated and out-of-range IDs
    are ignored. Output is local compressed token indices, padded with -1.
    """
    batch, next_n, heads, dim = q.shape
    rows = batch * next_n
    top_k = topk_indices.shape[1]
    capacity = candidate_blocks.shape[1] + triton.cdiv(top_k, candidate_block_size)
    blocks = torch.empty((rows, capacity), dtype=torch.int32, device=q.device)
    keep = torch.empty_like(blocks)
    lengths = torch.empty((rows,), dtype=torch.int32, device=q.device)
    logits = torch.empty(
        (rows, capacity * candidate_block_size), dtype=torch.float32, device=q.device
    )
    inplace = topk_indices.is_contiguous()
    indices = (
        topk_indices
        if inplace
        else torch.empty((rows, top_k), dtype=torch.int32, device=q.device)
    )
    assert q.dtype == torch.float8_e4m3fn and (heads, dim) == (32, 128)
    assert kv_cache.dtype == torch.uint8 and kv_cache.shape[-1] == dim + 4
    assert context_lens.shape in ((batch,), (batch, 1), (batch, next_n))
    assert candidate_blocks.dtype == torch.int32
    assert candidate_blocks.shape == (rows, 2048) and candidate_block_size == 8
    assert weights.shape == (rows, heads) and weights.dtype == torch.float32
    assert top_k == 512 and topk_indices.dtype == torch.int32
    assert max_model_len < 2**30
    lengths_2d = context_lens.ndim == 2 and context_lens.shape[1] == next_n
    length_stride = context_lens.stride(1) if lengths_2d else 0
    common = dict(
        NEXT_N=next_n,
        CAPACITY=capacity,
        CANDIDATE_SIZE=candidate_block_size,
        TOP_K=top_k,
        MAX_LEN=max_model_len,
        LENGTHS_2D=lengths_2d,
    )
    lut = _get_e4m3fn_bf16_lut(q.device)
    q_bf16 = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    _prepare_candidates[(rows,)](
        candidate_blocks,
        context_lens,
        blocks,
        keep,
        lengths,
        q.view(torch.uint8),
        q_bf16,
        lut,
        *q.stride(),
        *candidate_blocks.stride(),
        context_lens.stride(0),
        length_stride,
        CANDIDATES=candidate_blocks.shape[1],
        TILE=triton.next_power_of_2(candidate_blocks.shape[1]),
        num_warps=8,
        **common,
    )
    _candidate_paged_mqa[(rows, triton.cdiv(logits.shape[1], 128))](
        q_bf16,
        kv_cache,
        weights,
        lut,
        context_lens,
        block_tables,
        blocks,
        keep,
        logits,
        *q_bf16.stride(),
        kv_cache.stride(0),
        *weights.stride(),
        context_lens.stride(0),
        length_stride,
        *block_tables.stride(),
        HEADS=heads,
        DIM=dim,
        PAGE_SIZE=kv_cache.shape[1],
        BLOCK_H=max(16, triton.next_power_of_2(heads)),
        BLOCK_N=128,
        num_warps=4,
        num_stages=2,
        **common,
    )
    torch.ops._C.persistent_topk(
        logits, lengths, indices, workspace, top_k, logits.shape[1]
    )
    _restore_indices[(rows,)](
        indices,
        logits,
        blocks,
        context_lens,
        topk_indices,
        *topk_indices.stride(),
        context_lens.stride(0),
        length_stride,
        DENSE_FILTERED=rows > 32 or max_model_len <= 32768,
        INPLACE=inplace,
        num_warps=4,
        **common,
    )
