"""Functions shared between multiple tools."""

import torch
import torchaudio

def complex_norm(complex_tensor):
    # type: (Tensor) -> Tensor
    """Calculates the norm of PyTorch-style complex numbers."""
    return torch.norm(complex_tensor, 2, -1)


def combine_trace_windows(
    data_fx_windowed, trace_window_starts, trace_window_len, filter_len, data_fx_shape
):
    # type: (Tensor, Tensor, int, int, List[int]) -> Tuple[Tensor, Tensor]
    """Combines trace windows.

    Input data_fx_windowed: [batch_size, n_freqs, n_time_windows,
                             n_trace_windows, trace_window_len-filter_len, 2]
    Return data_fx: [batch_size, n_freqs, n_time_windows, n_traces, 2]
    """
    dtype = data_fx_windowed.dtype
    device = data_fx_windowed.device
    data_fx_combined = torch.zeros(
        data_fx_shape[0],
        data_fx_shape[1],
        data_fx_shape[2],
        data_fx_shape[3],
        data_fx_shape[4],
        device=device,
        dtype=dtype,
    )
    count = torch.zeros_like(data_fx_combined[..., 0])
    n_traces = data_fx_shape[-2]
    for trace_window_idx, trace_window_start in enumerate(trace_window_starts):
        trace_window_end = trace_window_start + trace_window_len
        trace_window = data_fx_windowed[..., trace_window_idx, :, :]
        data_fx_combined[
            ..., trace_window_start + filter_len : min(trace_window_end, n_traces), :
        ] += trace_window[
            ...,
            : min(trace_window_end, n_traces) - (trace_window_start + filter_len),
            :,
        ]
        count[
            ..., trace_window_start + filter_len : min(trace_window_end, n_traces)
        ] += 1
    return data_fx_combined, count


def extract_trace_windows(data_fx, trace_window_starts, trace_window_len):
    # type: (Tensor, Tensor, int) -> Tuple[Tensor, int]
    """Extracts windows of traces from the data.

    Returns:
        data_fx_trace_windowed: A Tensor of shape
                                [batch_size, n_freqs, n_time_windows,
                                 n_trace_windows, trace_window_len, 2]
        final_trace_window_len: An integer specifying the number of traces
                                in the final trace window
    """
    n_trace_windows = len(trace_window_starts)
    batch_size, n_freqs, n_time_windows, n_traces, _ = data_fx.shape
    dtype = data_fx.dtype
    device = data_fx.device
    data_fx_trace_windowed = torch.zeros(
        batch_size,
        n_freqs,
        n_time_windows,
        n_trace_windows,
        trace_window_len,
        2,
        device=device,
        dtype=dtype,
    )
    trace_window_len_actual = 0
    for trace_window_idx, trace_window_start in enumerate(trace_window_starts):
        trace_window_end = trace_window_start + trace_window_len
        trace_window = data_fx[
            ..., trace_window_start : min(trace_window_end, n_traces), :
        ]
        trace_window_len_actual = trace_window.shape[-2]
        data_fx_trace_windowed[
            ..., trace_window_idx, :trace_window_len_actual, :
        ] = trace_window
    # The final window may not be full - we need to store how many elements
    # are actually in it
    final_trace_window_len = trace_window_len_actual
    return data_fx_trace_windowed, final_trace_window_len


def restore_freq_window(data_fx, data_fx_denoised, min_freq, max_freq):
    # type: (Tensor, Tensor, float, float) -> Tensor
    """Replaces components between min and max freq with denoised values.

    Inputs:
        data_fx: [batch_size, n_freqs, n_time_windows, n_traces, 2]
        data_fx_denoised: [batch_size, n_freqs*(max_freq-min_freq),
                           n_time_windows, n_traces, 2]
        min_freq, max_freq: Floats specifying min and max Nyquist fractions

    Returns:
        data_fx: [batch_size, n_freqs, n_time_windows, n_traces, 2] after the
                 values between min and max freq have been replaced by those
                 in data_fx_denoised.
    """
    n_freqs = data_fx.shape[1]
    min_freq_idx = int(n_freqs * min_freq)
    max_freq_idx = int(n_freqs * max_freq)
    data_fx[:, min_freq_idx:max_freq_idx] = data_fx_denoised
    return data_fx


def extract_freq_window(data_fx, min_freq, max_freq):
    # type: (Tensor, float, float) -> Tensor
    """Extracts components corresponding to frequencies between min and max.

    Inputs:
        data_fx: [batch_size, n_freqs, n_time_windows, n_traces, 2]
        min_freq, max_freq: Floats specifying min and max Nyquist fractions

    Returns:
        data_fx_freq_windowed: [batch_size, n_freqs*(max_freq-min_freq),
                                n_time_windows, n_traces, 2]
    """
    n_freqs = data_fx.shape[1]
    min_freq_idx = int(n_freqs * min_freq)
    max_freq_idx = int(n_freqs * max_freq)
    return data_fx[:, min_freq_idx:max_freq_idx]


def inverse_fourier_transform_time(data_fx, time_window_len, n_times):
    # type: (Tensor, int, int) -> Tensor
    """Inverse Fourier transforms in time and combines overlapping windows.

    Inputs:
        data_fx: A [batch_size, n_freqs, n_time_windows, n_traces, 2] shape
                 Tensor containing the windowed and Fourier transformed
                 data
        time_window_len: An integer specifying the window length in the time
                         dimension to use when Fourier transforming the data.
        n_times: An integer specifying the length of the original data in the
                 time dimension.

    Returns:
        data: A [batch_size, n_traces, n_times] shape Tensor containing the
              data after inverse Fourier transforming and combining windows
    """
    # [batch_size, n_freqs, n_time_windows, n_traces, 2]
    # -> [batch_size, n_traces, n_freqs, n_time_windows, 2]
    # -> [batch_size * n_traces, n_freqs, n_time_windows, 2]
    # -> [batch_size * n_traces, n_times]
    # -> [batch_size, n_traces, n_times]
    batch_size, n_freqs, n_time_windows, n_traces, _ = data_fx.shape
    dtype = data_fx.dtype
    device = data_fx.device
    time_window = torch.hann_window(time_window_len, dtype=dtype, device=device)
    data_fx = data_fx.permute(0, 3, 1, 2, 4)
    data_fx = data_fx.reshape(batch_size * n_traces, n_freqs, n_time_windows, 2)
    data = torchaudio.functional.istft(
        data_fx,
        time_window_len,
        hop_length=time_window_len // 2,
        window=time_window,
        length=n_times,
    )
    return data.reshape(batch_size, n_traces, n_times)


def fourier_transform_time(data, time_window_len):
    # type: (Tensor, int) -> Tensor
    """Windows and Fourier transforms the data in time.

    Inputs:
        data: A [batch_size, n_traces, n_times] shape Tensor containing the data
        time_window_len: An integer specifying the window length in the time
                         dimension to use when Fourier transforming the data.

    Returns:
        data_fx: A [batch_size, n_freqs, n_time_windows, n_traces, 2] shape
                 Tensor containing the windowed and Fourier transformed
                 data

    """
    # Use the Short-Time Fourier Transform (STFT) to window data in time and
    # Fourier transform. This requires that the data is in 2D, with time
    # in the final dimension, so we need to combine the trace and batch
    # dimensions. To facilitate later steps of the process, we then shift
    # the trace dimension.
    # [batch_size, n_traces, n_times]
    # -> [batch_size * n_traces, n_times]
    # -> [batch_size * n_traces, n_freqs, n_time_windows, 2]
    # -> [batch_size, n_traces, n_freqs, n_time_windows, 2]
    # -> [batch_size, n_freqs, n_time_windows, n_traces, 2]
    batch_size, n_traces, n_times = data.shape
    dtype = data.dtype
    device = data.device
    time_window = torch.hann_window(time_window_len, dtype=dtype, device=device)
    data_fx = torch.stft(
        data.reshape(-1, n_times),
        time_window_len,
        hop_length=time_window_len // 2,
        window=time_window,
    )
    n_freqs, n_time_windows = data_fx.shape[1:3]
    data_fx = data_fx.reshape(batch_size, n_traces, n_freqs, n_time_windows, 2)
    data_fx = data_fx.permute(0, 2, 3, 1, 4)
    return data_fx

