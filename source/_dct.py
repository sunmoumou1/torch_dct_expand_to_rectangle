import torch
import torch.nn as nn
import numpy as np

class LinearDCT(nn.Linear):
    """
    Implements DCT as a linear layer that can handle 2D fields with unequal height and width.
    :param in_features: Input feature dimension
    :param type: Type of DCT used, such as 'dct', 'idct', etc.
    :param norm: Normalization parameter
    """
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # Initialize the weight matrix as a DCT or IDCT matrix
        I = torch.eye(self.N)
        if self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False  # Do not update weights


def apply_linear_2d(x, linear_layer_h, linear_layer_w):
    """
    Apply the LinearDCT layer to the last two dimensions for 2D DCT.
    :param x: Input signal, shape (B, C, H, W) or (B, H, W) anything...
    :param linear_layer_h: LinearDCT layer for the height direction
    :param linear_layer_w: LinearDCT layer for the width direction
    :return: DCT-transformed result
    """

    X1 = linear_layer_w(x)  
    X2 = linear_layer_h(X1.transpose(-1, -2)) 
    return X2.transpose(-1, -2).continuous()


def dct(x, norm=None):
    """
    1D DCT
    :param x: Input signal
    :param norm: Normalization option
    :return: DCT transformation result
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))
    # c表示complicated

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    1D inverse DCT
    :param X: Input signal
    :param norm: Normalization option
    :return: Inverse DCT transformation result
    """
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    # Note that 't' here stands for temporal
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    # This section represents complex number multiplication

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)
