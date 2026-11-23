"""Checkpoint-compatible expert execution. No capacity limit or token dropping."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _grouped_linear(
    X,
    W,
    BIAS,
    Y,
    OFF,
    TILES,
    E: tl.constexpr,
    N: tl.constexpr,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    WSN: tl.constexpr,
    WSK: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 64,
    BK: tl.constexpr = 32,
):
    pid = tl.program_id(0)
    ends = tl.load(
        TILES + tl.arange(0, triton.next_power_of_2(E)),
        mask=tl.arange(0, triton.next_power_of_2(E)) < E,
        other=2147483647,
    )
    expert = tl.sum((pid >= ends).to(tl.int32), 0)
    if expert < E:
        tile_start = tl.load(TILES + expert - 1) if expert > 0 else 0
        begin = tl.load(OFF + expert)
        end = tl.load(OFF + expert + 1)
        rows = begin + (pid - tile_start) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in range(triton.cdiv(CIN, BK)):
            k = block * BK + ks
            a = tl.load(
                X + rows[:, None] * CIN + k[None, :],
                mask=(rows[:, None] < end) & (k[None, :] < CIN),
                other=0,
            )
            b = tl.load(
                W + expert * CIN * COUT + cols[None, :] * WSN + k[:, None] * WSK,
                mask=(cols[None, :] < COUT) & (k[:, None] < CIN),
                other=0,
            )
            acc += tl.dot(a, b, input_precision="ieee")
        if BIAS is not None:
            bias = tl.load(BIAS + expert * COUT + cols, mask=cols < COUT, other=0)
            acc += bias[None, :].to(tl.float32)
        tl.store(
            Y + rows[:, None] * COUT + cols[None, :],
            acc,
            mask=(rows[:, None] < end) & (cols[None, :] < COUT),
        )


@triton.jit
def _weight_grad(
    X,
    DY,
    DW,
    OFF,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 32,
    BK: tl.constexpr = 32,
):
    expert = tl.program_id(2)
    outs = tl.program_id(0) * BM + tl.arange(0, BM)
    ins = tl.program_id(1) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + expert)
    end = tl.load(OFF + expert + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(begin, end, BK):
        rows = start + tl.arange(0, BK)
        a = tl.load(
            DY + rows[None, :] * COUT + outs[:, None],
            mask=(rows[None, :] < end) & (outs[:, None] < COUT),
            other=0,
        )
        b = tl.load(
            X + rows[:, None] * CIN + ins[None, :],
            mask=(rows[:, None] < end) & (ins[None, :] < CIN),
            other=0,
        )
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(
        DW + expert * COUT * CIN + outs[:, None] * CIN + ins[None, :],
        acc,
        mask=(outs[:, None] < COUT) & (ins[None, :] < CIN),
    )


class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, offsets, tiles, expert_ids):
        x = x.contiguous()
        w = w.contiguous()
        b = b.contiguous()
        e, cout, cin = w.shape
        y = torch.empty((len(x), cout), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(len(x), 32) + e, triton.cdiv(cout, 64))
        _grouped_linear[grid](x, w, b, y, offsets, tiles, e, len(x), cin, cout, cin, 1)
        ctx.save_for_backward(x, w, offsets, tiles, expert_ids)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, offsets, tiles, expert_ids = ctx.saved_tensors
        dy = dy.contiguous()
        e, cout, cin = w.shape
        dx = torch.empty_like(x)
        dw = torch.empty_like(w)
        _grouped_linear[(triton.cdiv(len(x), 32) + e, triton.cdiv(cin, 64))](
            dy, w, None, dx, offsets, tiles, e, len(x), cout, cin, 1, cin
        )
        _weight_grad[(triton.cdiv(cout, 32), triton.cdiv(cin, 32), e)](
            x, dy, dw, offsets, cin, cout
        )
        db = torch.zeros((e, cout), device=dy.device, dtype=torch.float32)
        db.index_add_(0, expert_ids, dy.float())
        return dx, dw, db.to(w.dtype), None, None, None


def _parameters(bank, dtype):
    # Stack through autograd; original Parameter objects/state_dict keys remain.
    return tuple(
        torch.stack([getattr(expert[layer], field) for expert in bank.experts]).to(
            dtype
        )
        for layer, field in ((0, "weight"), (0, "bias"), (3, "weight"), (3, "bias"))
    )


def sparse_experts(bank, features, weights, k):
    shape = features.shape
    x = features.reshape(-1, shape[-1])
    # Preserve ascending expert order in the final weighted sum.
    selected = (
        weights.reshape(-1, bank.num_experts)
        .topk(k, dim=-1)
        .indices.sort(dim=-1)
        .values
    )
    selected_weights = weights.reshape(-1, bank.num_experts).gather(1, selected)
    ids = selected.flatten()
    order = ids.argsort(stable=True)
    sorted_ids = ids.index_select(0, order)
    counts = torch.zeros(bank.num_experts, device=x.device, dtype=torch.int32)
    counts.scatter_add_(0, sorted_ids, torch.ones_like(sorted_ids, dtype=torch.int32))
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)])
    tiles = ((counts + 31) // 32).cumsum(0, dtype=torch.int32)
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(len(order), device=order.device))
    dtype = (
        torch.get_autocast_dtype("cuda")
        if torch.is_autocast_enabled("cuda")
        else features.dtype
    )
    packed = x.index_select(0, order // k).to(dtype)

    def layer(value, index):
        modules = [expert[index] for expert in bank.experts]
        bias = torch.stack([m.bias for m in modules]).to(dtype)
        if getattr(modules[0], "rank", None) is not None:
            right = torch.stack([m.right for m in modules]).to(dtype)
            left = torch.stack([m.left for m in modules]).to(dtype)
            zero = torch.zeros(
                (bank.num_experts, right.shape[1]), device=value.device, dtype=dtype
            )
            value = _GroupedLinear.apply(value, right, zero, offsets, tiles, sorted_ids)
            return _GroupedLinear.apply(value, left, bias, offsets, tiles, sorted_ids)
        weight = torch.stack([m.weight for m in modules]).to(dtype)
        return _GroupedLinear.apply(value, weight, bias, offsets, tiles, sorted_ids)

    hidden = F.silu(layer(packed, 0))
    values = layer(hidden, 3)
    values = values.index_select(0, inverse).reshape(len(x), k, shape[-1])
    residual = torch.zeros_like(
        x, dtype=torch.promote_types(features.dtype, weights.dtype)
    )
    for slot in range(k):
        residual = residual + values[:, slot] * selected_weights[:, slot, None]
    return residual.reshape(shape)


def dense_experts(bank, features, weights):
    shape = features.shape
    x = features.reshape(-1, shape[-1])
    dtype = (
        torch.get_autocast_dtype("cuda")
        if torch.is_autocast_enabled("cuda")
        else features.dtype
    )
    w1, b1, w2, b2 = _parameters(bank, dtype)
    x = x.to(dtype)
    # Use addmm's fused bias rounding, rather than a separate BF16 bias add.
    hidden = torch.baddbmm(
        b1[:, None, :], x[None].expand(bank.num_experts, -1, -1), w1.transpose(1, 2)
    )
    hidden = F.silu(hidden)
    values = torch.baddbmm(b2[:, None, :], hidden, w2.transpose(1, 2))
    scales = weights.reshape(-1, bank.num_experts).transpose(0, 1)
    residual = torch.zeros_like(
        x, dtype=torch.promote_types(features.dtype, weights.dtype)
    )
    for e in range(bank.num_experts):
        residual = residual + values[e] * scales[e, :, None]
    return residual.reshape(shape)
