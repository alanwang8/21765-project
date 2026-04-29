"""Flash Attention operations."""

from ..autograd import Op, Tensor, TensorOp, TensorTuple, TensorTupleOp
from .. import backend_ndarray as nd


class FlashAttention(TensorOp):
    def __init__(self, causal: bool = False, block_m: int = 128, block_n: int = None):
        self.causal = causal
        self.block_m = block_m
        self.block_n = block_n if block_n is not None else block_m

    def compute(self, Q, K, V):
        if Q.device.name != 'cuda':
            raise NotImplementedError("FlashAttention only supported on CUDA.")

        Q_c, K_c, V_c = Q.compact(), K.compact(), V.compact()
        batch_size, num_heads, seq_len, head_dim = Q_c.shape

        out = nd.NDArray.make(Q_c.shape, device=Q.device)
        stats_shape = (batch_size, num_heads, seq_len)
        m = nd.NDArray.make(stats_shape, device=Q.device)
        l = nd.NDArray.make(stats_shape, device=Q.device)

        Q.device.flash_attention_forward(
            Q_c._handle, K_c._handle, V_c._handle,
            out._handle, m._handle, l._handle,
            batch_size, num_heads, seq_len, head_dim,
            self.block_m, self.block_n, self.causal
        )

        out._cached_m = m
        out._cached_l = l
        return out

    def gradient(self, out_grad, node):
        q, k, v = node.inputs
        grads = flash_attention_grad(out_grad, q, k, v, node,
                                     self.causal, self.block_m, self.block_n)
        return tuple(grads)


class FlashAttentionGrad(TensorTupleOp):
    def __init__(self, causal, block_m, block_n):
        self.causal = causal
        self.block_m = block_m
        self.block_n = block_n

    def compute(self, dO, Q, K, V, O_ref):
        dO_c = dO.compact()
        Q_c, K_c, V_c = Q.compact(), K.compact(), V.compact()

        if not hasattr(O_ref, '_cached_m') or not hasattr(O_ref, '_cached_l'):
            raise RuntimeError(
                "Backward pass failed: m/l stats not found on forward output."
            )

        m, l = O_ref._cached_m, O_ref._cached_l
        batch_size, num_heads, seq_len, head_dim = Q_c.shape

        dQ = nd.NDArray.make(Q_c.shape, device=Q.device)
        dK = nd.NDArray.make(K_c.shape, device=Q.device)
        dV = nd.NDArray.make(V_c.shape, device=Q.device)

        Q.device.flash_attention_backward(
            dO_c._handle,
            Q_c._handle, K_c._handle, V_c._handle,
            m._handle, l._handle,
            dQ._handle, dK._handle, dV._handle,
            batch_size, num_heads, seq_len, head_dim,
            self.block_m, self.block_n, self.causal
        )

        return dQ, dK, dV


def flash_attention(Q, K, V, causal: bool = False, block_m: int = 128, block_n: int = None):
    return FlashAttention(causal=causal, block_m=block_m, block_n=block_n)(Q, K, V)


def flash_attention_grad(dO, Q, K, V, O, causal, block_m, block_n):
    return FlashAttentionGrad(causal, block_m, block_n)(dO, Q, K, V, O)
