import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    # get last 2 dim
    T, C = query.shape[-2], query.shape[-1]
    scale = mx.divide(1, mx.sqrt(C)) if scale is None else scale
    tmp = mx.multiply(mx.matmul(query, mx.swapaxes(key, -1, -2)), scale)
    tmp = mx.add(tmp, mask) if mask is not None else tmp
    tmp = softmax(tmp, -1)
    return mx.matmul(tmp, value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.num_heads = num_heads
        self.hidden_size = hidden_size

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        # B * T * hidden_dim
        # hidden_dim = num_head * head_dim
        N, L, E = query.shape
        head_dim = E // self.num_heads
        assert head_dim * self.num_heads == self.hidden_size
        # get K, Q, V
        # B, T, C -> B, T, num_head, head_dim -> B, num_head, T, head_dim
        Q = mx.reshape(mx.matmul(query, mx.transpose(self.wq)), (N, L, self.num_heads, head_dim))
        K = mx.reshape(mx.matmul(key, mx.transpose(self.wk)), (N, L, self.num_heads, head_dim))
        V = mx.reshape(mx.matmul(value, mx.transpose(self.wv)), (N, L, self.num_heads, head_dim))
        Q = mx.swapaxes(Q, 1, 2)
        K = mx.swapaxes(K, 1, 2)
        V = mx.swapaxes(V, 1, 2)

        # B, num_head, T, head_dim -> B, T, num_head, head_dim -> B, T, C
        attention = scaled_dot_product_attention_simple(Q, K, V, None, mask)
        attention = mx.reshape(mx.swapaxes(attention, 1, 2), (N, L, E))
        return linear(attention, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
