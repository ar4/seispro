"""Automatic Gain Control (AGC)"""
import torch
import numpy as np

def agc(data, time_window_side_len=64):
    # types: (Tensor, int) -> Tuple[Tensor, Tensor]
    """Applies Automatic Gain Control (AGC).

    AGC is a commonly used process to reduce the amplitude range of data
    or an image. In this implementation, it is based on the smoothed
    absolute value of each trace. Smoothing is performing by convolving
    the absolute value with a boxcar of length 2*time_window_side_len+1.
    Mirror/reflection padding is applied to the edges.

    Inputs:
        data: A Tensor of shape [batch_size, n_traces, n_times]
        time_window_side_len: An integer controlling the number of samples in
                              the final dimension of the data (usually time
                              or depth) to use when calculating the scaling.
                              The number used will be 2*trace_window_len+1.

    Returns:
        data: The input data after AGC
        scaling: A Tensor of the same shape as the data, containing the
                 scaling applied to each sample. This can be used to
                 undo AGC (by dividing the data by the scaling).
    """
    _, _, n_times = data.shape
    if not np.issubdtype(type(time_window_side_len), np.integer):
        raise RuntimeError("time_window_side_len must be an integer")
    if not 1 <= 2*time_window_side_len+1 <= n_times:
        raise RuntimeError("2*time_window_side_len+1 must be in [1, n_times]")
    return agc_jit(data, time_window_side_len)

#@torch.jit.script
def agc_jit(data, time_window_side_len):
    # types: (Tensor, int) -> Tuple[Tensor, Tensor]
    """JIT-compiled function without error checking.

    raise seems to not be supported by Torchscript, so error checking
    is done before calling this JIT-compiled function.
    """
    batch_size, n_traces, n_times = data.shape
    dtype = data.dtype
    device = data.device
    data_abs = data.abs()
    # PyTorch 1D padding and convolution require an input shape
    # [batch_size, channels, conv_dim] for the data,
    # and [in_chans, out_chans, kernel_size] for the kernel
    data_abs = data_abs.reshape(batch_size * n_traces, 1, n_times)
    # Apply reflection padding to the top and bottom of the traces
    # to try to reduce edge effects when convolving
    data_abs_padded = torch.nn.functional.pad(data_abs,
                                              (time_window_side_len,
                                               time_window_side_len),
                                              mode='reflect')
    kernel = (torch.ones(1, 1, 2*time_window_side_len+1, dtype=dtype,
                         device=device) / (2*time_window_side_len+1))
    scaling = torch.nn.functional.conv1d(data_abs_padded, kernel)
    # Remove the unused channel dimension and uncombine first two dimensions
    scaling = scaling.reshape(batch_size, n_traces, n_times)
    # Take the reciprocal, so the scaling is applied by multiplication. To
    # avoid the possibility of division by zero, a small value is added
    scaling = 1 / (scaling + scaling.max()*1e-7)
    return data * scaling, scaling
