# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Device-length-aware candidate selection without sorting or CTA rendezvous."""

import torch

from vllm.triton_utils import tl, triton

_HIST_TILE = 4096
_LONG_ROW = 8192


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _ordered_key(value):
    bits = value.to(tl.uint32, bitcast=True)
    key = tl.where((bits & 0x80000000) != 0, bits ^ 0xFFFFFFFF, bits ^ 0x80000000)
    return tl.where(value != value, 0xFFFFFFFF, key).to(tl.uint32)


@triton.jit
def _coarse_key(value):
    bits = value.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
    key = tl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits ^ 0x8000)
    return tl.where(value != value, 511, key >> 7).to(tl.int32)


@triton.jit(do_not_specialize=["width", "nblocks"])
def _ragged_scores_kernel(
    logits,
    starts,
    ends,
    scores,
    lengths,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    count = tl.cdiv(tl.maximum(tl.minimum(end, width) - start, 0), BLOCK_SIZE)
    newest = (end - start - 1) // BLOCK_SIZE
    pinned = (end > start) & (newest >= 0) & (newest < nblocks)
    # The baseline can pin a block beyond the allocated logits' valid columns.
    count = tl.minimum(tl.maximum(count, tl.where(pinned, newest + 1, 0)), nblocks)
    if tl.program_id(1) == 0:
        tl.store(lengths + row, count)
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    for tile in range(tl.program_id(1), tl.cdiv(count, TILE), tl.num_programs(1)):
        blocks = tile * TILE + tl.arange(0, TILE)
        cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
        values = tl.load(
            logits + row * stride_row + cols * stride_col,
            (blocks[:, None] < count)
            & (offsets[None, :] < BLOCK_SIZE)
            & (cols < end)
            & (cols < width),
            other=-float("inf"),
        ).to(tl.float32)
        reduced = tl.reduce(values, 1, _max_with_nan)
        reduced = tl.where(pinned & (blocks == newest), float("inf"), reduced)
        tl.store(scores + row * nblocks + blocks, reduced, blocks < count)


@triton.jit(do_not_specialize=["nblocks"])
def _coarse_histogram_kernel(
    scores,
    lengths,
    histograms,
    nblocks,
    SHARDS: tl.constexpr,
    TILE: tl.constexpr,
    LONG_ROW: tl.constexpr,
    WARP_AGG: tl.constexpr = True,
):
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(lengths + row)
    if (length > LONG_ROW) & (tl.program_id(1) * TILE < length):
        cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
        value = tl.load(scores + row * nblocks + cols, cols < length, other=0).to(
            tl.float32
        )
        bins = tl.arange(0, 512)
        histogram_ptr = histograms + (row * SHARDS + tl.program_id(1)) * 512
        if WARP_AGG:
            tl.store(histogram_ptr + bins, 0)
            tl.debug_barrier()
            key = tl.where(cols < length, _coarse_key(value), 512)
            weight = tl.inline_asm_elementwise(
                """{
                    .reg .b32 peers, lane, leader, count;
                    .reg .pred first;
                    mov.u32 lane, %laneid;
                    match.any.sync.b32 peers, $1, 0xffffffff;
                    brev.b32 leader, peers;
                    clz.b32 leader, leader;
                    setp.eq.u32 first, lane, leader;
                    popc.b32 count, peers;
                    selp.u32 $0, count, 0, first;
                }""",
                constraints="=r,r",
                args=[key],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            tl.atomic_add(
                histogram_ptr + key,
                weight,
                (cols < length) & (weight > 0),
                sem="relaxed",
            )
        else:
            histogram = tl.histogram(_coarse_key(value), 512, mask=cols < length)
            tl.store(histogram_ptr + bins, histogram)


@triton.jit
def _coarse_threshold_kernel(
    histograms,
    lengths,
    state,
    K: tl.constexpr,
    SHARDS: tl.constexpr,
    LONG_ROW: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if tl.load(lengths + row) > LONG_ROW:
        bins = tl.arange(0, 512)
        histogram = tl.full((512,), 0, tl.int32)
        for shard in range(tl.cdiv(tl.load(lengths + row), 4096)):
            histogram += tl.load(histograms + (row * SHARDS + shard) * 512 + bins)
        suffix = tl.cumsum(histogram, reverse=True)
        pivot = tl.max(tl.where(suffix >= K, bins, 0), 0)
        above = tl.sum(tl.where(bins > pivot, histogram, 0), 0)
        tl.store(state + row * 5, pivot)
        tl.store(state + row * 5 + 1, K - above)
        tl.store(state + row * 5 + 2, 0)
        tl.store(state + row * 5 + 3, 0xFFFFFFFF)
        tl.store(state + row * 5 + 4, 0)


@triton.jit(do_not_specialize=["nblocks"])
def _compact_boundary_kernel(
    scores,
    lengths,
    state,
    boundary,
    nblocks,
    TILE: tl.constexpr,
    LONG_ROW: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(lengths + row)
    if (length > LONG_ROW) & (tl.program_id(1) * TILE < length):
        cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
        value = tl.load(scores + row * nblocks + cols, cols < length, other=0).to(
            tl.float32
        )
        keys = _ordered_key(value)
        pivot = tl.load(state + row * 5).to(tl.uint32)
        match = (cols < length) & (_coarse_key(value) == pivot)
        count = tl.sum(match.to(tl.int32), 0)
        base = tl.atomic_add(state + row * 5 + 2, count.to(tl.int64), sem="relaxed")
        rank = tl.cumsum(match.to(tl.int32)) - 1
        tl.store(
            boundary + row * nblocks + base + rank,
            keys.to(tl.int32, bitcast=True),
            match,
        )
        smallest = tl.min(tl.where(match, keys, 0xFFFFFFFF), 0).to(tl.int64)
        largest = tl.max(tl.where(match, keys, 0), 0).to(tl.int64)
        tl.atomic_min(state + row * 5 + 3, smallest, sem="relaxed")
        tl.atomic_max(state + row * 5 + 4, largest, sem="relaxed")


@triton.jit(do_not_specialize=["nblocks"])
def _radix_candidates_kernel(
    scores,
    lengths,
    out,
    nblocks,
    out_stride_row,
    out_stride_col,
    state,
    boundary,
    selection,
    K: tl.constexpr,
    TILE: tl.constexpr,
    LONG_ROW: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(lengths + row)
    offsets = tl.arange(0, TILE)
    if length <= K:
        for tile in range(tl.cdiv(K, TILE)):
            cols = tile * TILE + offsets
            value = tl.load(
                scores + row * nblocks + cols, cols < length, other=-float("inf")
            )
            tl.store(
                out + row * out_stride_row + cols * out_stride_col,
                tl.where((cols < length) & (value > -float("inf")), cols, -1),
                cols < K,
            )
    else:
        bins = tl.arange(0, 256)
        prefix = tl.full((), 0, tl.uint32)
        mask = tl.full((), 0, tl.uint32)
        remaining = tl.full((), K, tl.int32)
        resolved = tl.full((), False, tl.int1)
        shift = 24
        count = length
        if length > LONG_ROW:
            remaining = tl.load(state + row * 5 + 1).to(tl.int32)
            count = tl.load(state + row * 5 + 2).to(tl.int32)
            smallest = tl.load(state + row * 5 + 3).to(tl.uint32)
            largest = tl.load(state + row * 5 + 4).to(tl.uint32)
            if smallest == largest:
                prefix = smallest
                shift = -1
        while (shift >= 0) & ~resolved:
            histogram = tl.full((256,), 0, tl.int32)
            for tile in range(tl.cdiv(count, TILE)):
                cols = tile * TILE + offsets
                if length > LONG_ROW:
                    key = tl.load(
                        boundary + row * nblocks + cols, cols < count, other=0
                    ).to(tl.uint32, bitcast=True)
                else:
                    value = tl.load(
                        scores + row * nblocks + cols, cols < count, other=0
                    ).to(tl.float32)
                    key = _ordered_key(value)
                active = (cols < count) & ((key & mask) == prefix)
                histogram += tl.histogram(
                    ((key >> shift) & 255).to(tl.int32), 256, mask=active
                )
            suffix = tl.cumsum(histogram, reverse=True)
            pivot = tl.max(tl.where(suffix >= remaining, bins, 0), 0)
            above = tl.sum(tl.where(bins > pivot, histogram, 0), 0)
            equal = tl.sum(tl.where(bins == pivot, histogram, 0), 0)
            remaining -= above
            prefix |= pivot.to(tl.uint32) << shift
            mask |= tl.full((), 255, tl.uint32) << shift
            # Compacted FP16 buckets need a full FP32 cutoff: an early partial
            # prefix could also admit keys from a lower, unselected coarse bin.
            resolved = (remaining == equal) & (length <= LONG_ROW)
            shift = tl.where(shift == 4, 0, shift - 8)

        if length > LONG_ROW:
            tl.store(selection + row * 2, prefix.to(tl.int64))
            tl.store(selection + row * 2 + 1, tl.where(resolved, K, remaining))
        else:
            written = tl.full((), 0, tl.int32)
            ties_seen = tl.full((), 0, tl.int32)
            for tile in range(tl.cdiv(length, TILE)):
                cols = tile * TILE + offsets
                value = tl.load(
                    scores + row * nblocks + cols, cols < length, other=0
                ).to(tl.float32)
                key = _ordered_key(value)
                tie = (cols < length) & (key == prefix)
                tie_rank = tl.cumsum(tie.to(tl.int32)) + ties_seen
                chosen = (cols < length) & (
                    (key > prefix) | (tie & (resolved | (tie_rank <= remaining)))
                )
                rank = tl.cumsum(chosen.to(tl.int32)) - 1 + written
                tl.store(
                    out + row * out_stride_row + rank * out_stride_col,
                    tl.where(value > -float("inf"), cols, -1),
                    chosen & (rank < K),
                )
                written += tl.sum(chosen.to(tl.int32), 0)
                ties_seen += tl.sum(tie.to(tl.int32), 0)


@triton.jit(do_not_specialize=["nblocks"])
def _selection_counts_kernel(
    scores,
    lengths,
    selection,
    counts,
    nblocks,
    K: tl.constexpr,
    TILES: tl.constexpr,
    TILE: tl.constexpr,
    LONG_ROW: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(lengths + row)
    if (length > LONG_ROW) & (length > K) & (tl.program_id(1) * TILE < length):
        cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
        value = tl.load(scores + row * nblocks + cols, cols < length, other=0).to(
            tl.float32
        )
        keys = _ordered_key(value)
        pivot = tl.load(selection + row * 2).to(tl.uint32)
        above = tl.sum(((cols < length) & (keys > pivot)).to(tl.int32), 0)
        equal = tl.sum(((cols < length) & (keys == pivot)).to(tl.int32), 0)
        tl.store(counts + (row * TILES + tl.program_id(1)) * 2, above)
        tl.store(counts + (row * TILES + tl.program_id(1)) * 2 + 1, equal)


@triton.jit
def _selection_prefix_kernel(
    lengths,
    selection,
    counts,
    offsets,
    K: tl.constexpr,
    TILES: tl.constexpr,
    TILE: tl.constexpr,
    LONG_ROW: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(lengths + row)
    if (length > LONG_ROW) & (length > K):
        tiles = tl.arange(0, triton.next_power_of_2(TILES))
        valid = tiles < tl.cdiv(length, TILE)
        above = tl.load(counts + (row * TILES + tiles) * 2, valid, other=0)
        equal = tl.load(counts + (row * TILES + tiles) * 2 + 1, valid, other=0)
        previous_ties = tl.cumsum(equal) - equal
        quota = tl.minimum(
            tl.maximum(tl.load(selection + row * 2 + 1) - previous_ties, 0), equal
        )
        selected = above + quota
        base = tl.cumsum(selected) - selected
        tl.store(offsets + (row * TILES + tiles) * 2, base, valid)
        tl.store(offsets + (row * TILES + tiles) * 2 + 1, quota, valid)


@triton.jit(do_not_specialize=["nblocks"])
def _selection_scatter_kernel(
    scores,
    lengths,
    selection,
    offsets,
    out,
    nblocks,
    out_stride_row,
    out_stride_col,
    K: tl.constexpr,
    TILES: tl.constexpr,
    TILE: tl.constexpr,
    LONG_ROW: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(lengths + row)
    if (length > LONG_ROW) & (length > K) & (tl.program_id(1) * TILE < length):
        cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
        value = tl.load(scores + row * nblocks + cols, cols < length, other=0).to(
            tl.float32
        )
        keys = _ordered_key(value)
        pivot = tl.load(selection + row * 2).to(tl.uint32)
        base = tl.load(offsets + (row * TILES + tl.program_id(1)) * 2)
        quota = tl.load(offsets + (row * TILES + tl.program_id(1)) * 2 + 1)
        equal = (cols < length) & (keys == pivot)
        chosen = (cols < length) & (
            (keys > pivot) | (equal & (tl.cumsum(equal.to(tl.int32)) <= quota))
        )
        rank = base + tl.cumsum(chosen.to(tl.int32)) - 1
        tl.store(
            out + row * out_stride_row + rank * out_stride_col,
            tl.where(value > -float("inf"), cols, -1),
            chosen & (rank < K),
        )


def select_candidate_blocks_radix(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
) -> None:
    """Select the same candidate set; publish IDs in ascending input order.

    Candidate IDs feed a membership mask, so their order has no numerical effect.
    Equal boundary scores choose the lowest block IDs, as CUDA gather-topk does.
    NaNs consume selection slots and then become -1, matching the baseline store.
    """
    rows, width = logits.shape
    nblocks = triton.cdiv(width, block_size)
    scores = logits.new_empty((rows, nblocks))
    lengths = torch.empty(rows, device=logits.device, dtype=torch.int32)
    shards = triton.cdiv(nblocks, _HIST_TILE)
    histograms = torch.empty(
        (rows, shards, 512), device=logits.device, dtype=torch.int32
    )
    state = torch.empty((rows, 5), device=logits.device, dtype=torch.int64)
    boundary = torch.empty((rows, nblocks), device=logits.device, dtype=torch.int32)
    tiles = triton.cdiv(nblocks, 1024)
    counts = torch.empty((rows, tiles, 2), device=logits.device, dtype=torch.int32)
    offsets = torch.empty_like(counts)
    selection = torch.empty((rows, 2), device=logits.device, dtype=torch.int64)
    _ragged_scores_kernel[(rows, min(triton.cdiv(nblocks, 128), 32))](
        logits,
        row_ks,
        row_ke,
        scores,
        lengths,
        *logits.stride(),
        row_ks.stride(0) if row_ks is not None else 0,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        128,
    )
    _coarse_histogram_kernel[(rows, shards)](
        scores,
        lengths,
        histograms,
        nblocks,
        shards,
        _HIST_TILE,
        _LONG_ROW,
    )
    _coarse_threshold_kernel[(rows,)](
        histograms,
        lengths,
        state,
        topk_blocks,
        shards,
        _LONG_ROW,
    )
    _compact_boundary_kernel[(rows, triton.cdiv(nblocks, 1024))](
        scores,
        lengths,
        state,
        boundary,
        nblocks,
        1024,
        _LONG_ROW,
    )
    _radix_candidates_kernel[(rows,)](
        scores,
        lengths,
        out,
        nblocks,
        *out.stride(),
        state,
        boundary,
        selection,
        topk_blocks,
        1024,
        _LONG_ROW,
        num_warps=4,
    )
    _selection_counts_kernel[(rows, tiles)](
        scores,
        lengths,
        selection,
        counts,
        nblocks,
        topk_blocks,
        tiles,
        1024,
        _LONG_ROW,
    )
    _selection_prefix_kernel[(rows,)](
        lengths,
        selection,
        counts,
        offsets,
        topk_blocks,
        tiles,
        1024,
        _LONG_ROW,
    )
    _selection_scatter_kernel[(rows, tiles)](
        scores,
        lengths,
        selection,
        offsets,
        out,
        nblocks,
        *out.stride(),
        topk_blocks,
        tiles,
        1024,
        _LONG_ROW,
    )
