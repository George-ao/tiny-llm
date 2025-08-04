# what I have learned 0.0
# it takes me a long time to know that I cannot do [][][] on mlx, numpy or pytorch !
# output[batch][pos][2*i+1] = x[batch][pos][1] * used_cos_freq[pos][i] + x[batch][pos][0] * used_sin_freq[pos][i]
import mlx.core as mx
class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dim = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        self.sin_freq, self.cos_freq = self.pre_compute_frequency()

    def pre_compute_frequency(self,):
        w_i = 1.0 / mx.power(self.base, mx.arange(0, self.dim, 2) / self.dim)
        # seq_len, dim
        pos = mx.arange(0, self.seq_len)
        sin_freq = mx.sin(mx.outer(pos, w_i))
        cos_freq = mx.cos(mx.outer(pos, w_i))
        return sin_freq, cos_freq
    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, L, H, D = x.shape
        assert D <= self.dim
        assert L <= self.seq_len
        assert D % 2 ==0

        if self.traditional:
            x = mx.swapaxes(x, 1, 2)
            x = mx.reshape(x, (-1, L, D))
            # L, D//2
            used_sin_freq = self.sin_freq[offset, :] if offset else self.sin_freq[:L,:]
            used_cos_freq = self.cos_freq[offset, :] if offset else self.cos_freq[:L,:]
            # broadcast B, L, D//2
            used_cos_freq = used_cos_freq[None, ...]
            used_sin_freq = used_sin_freq[None, ...]
            # B, L, D -> B, L, D//2
            even_x = x[..., 0::2]
            odd_x = x[..., 1::2]
            even_out = mx.multiply(even_x, used_cos_freq) - mx.multiply(odd_x, used_sin_freq)
            odd_out = mx.multiply(even_x, used_sin_freq) + mx.multiply(odd_x, used_cos_freq)
            out = mx.stack((even_out, odd_out), -1)
            out = mx.reshape(out, (N, H, L ,D))
            return mx.swapaxes(out, 1,2)
        else:
            x = mx.swapaxes(x, 1, 2)
            x = mx.reshape(x, (-1, L, D))
            used_sin_freq = self.sin_freq[offset, :] if offset else self.sin_freq[:L,:]
            used_cos_freq = self.cos_freq[offset, :] if offset else self.cos_freq[:L,:]
            # broadcast B, L, D
            used_cos_freq = used_cos_freq[None, ...]
            used_sin_freq = used_sin_freq[None, ...]
            # B, L, D -> B, L, D//2
            half_dim = D//2
            x_1 = x[..., :half_dim]
            x_2 = x[..., half_dim:]
            even_out = mx.multiply(x_1, used_cos_freq) - mx.multiply(x_2, used_sin_freq)
            odd_out = mx.multiply(x_1, used_sin_freq) + mx.multiply(x_2, used_cos_freq)
            out = mx.concatenate((even_out, odd_out), -1)
            out = mx.reshape(out, (N, H, L ,D))
            return mx.swapaxes(out, 1,2)
        
