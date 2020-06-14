"""Time-frequency clipping"""
import torch
import numpy as np
from seispro.shared import combine_trace_windows, extract_trace_windows, restore_freq_window, extract_freq_window, inverse_fourier_transform_time, fourier_transform_time, complex_norm

def tfclip(data, trace_window_len=5, time_window_len=512, min_freq=0.0, max_freq=1.0, clip_factor=1.5):
    # type: (Tensor, int, int, float, float, float) -> Tensor
    """Clips fx values above a multiple of the median within windows.

    The data is windowed in trace and time dimensions, and then
    Fourier transformed in time. The median magnitude of these transformed
    values across traces are calculated, and any value that are larger
    than clip_factor * median are clipped to the median.

    This tool is useful for attenuating strong bursts of noise that only
    occur on a few traces.

    Inputs:
        data: A [batch_size, n_traces, n_times] shape Tensor containing the data
        trace_window_len: An integer specifying the window length in the trace
                          dimension to use when calculating the median.
                          It must be at least 3. Default 5.
        time_window_len: An integer specifying the window length in the time
                         dimension to use when Fourier transforming the data.
                         Default 512.
        min_freq: A float specifying the minimum fraction of the
                  Nyquist frequency to apply the tool to. Default: 0.
        max_freq: A float specifying the maximum fraction of the
                  Nyquist frequency to apply the tool to. Default: 1.
        clip_factor: The maximum alloable multiple of the median magnitude.
                     Values above clip_factor * median are clipped.

    Returns:
        data: A Tensor of the same shape as the input, after filtering.
    """
    batch_size, n_traces, n_times = data.shape
    if n_traces < 3:
        raise RuntimeError("number of traces must be >= 3")
    if not np.issubdtype(type(trace_window_len), np.integer):
        raise RuntimeError("trace_window_len must be an integer")
    if not 3 <= trace_window_len <= n_traces:
        raise RuntimeError("trace_window_len must be in [3, n_traces]")
    if not np.issubdtype(type(time_window_len), np.integer):
        raise RuntimeError("time_window_len must be an integer")
    if not 2 <= time_window_len <= n_times:
        raise RuntimeError("time_window_len must be in [2, n_times]")
    if not 0.0 <= min_freq < 1.0:
        raise RuntimeError("min_freq must be in [0.0, 1.0)")
    if not 0.0 < max_freq <= 1.0:
        raise RuntimeError("max_freq must be in (0.0, 1.0]")
    if not 0.0 < clip_factor:
        raise RuntimeError("clip_factor must be greater than 0.0")
    return tfclip_jit(
        data, trace_window_len, time_window_len, min_freq, max_freq, clip_factor
    )

@torch.jit.script
def tfclip_jit(data, trace_window_len, time_window_len, min_freq, max_freq, clip_factor):
    # type: (Tensor, int, int, float, float, float) -> Tensor
    """JIT-compiled function without error checking.

    raise seems to not be supported by Torchscript, so error checking
    is done before calling this JIT-compiled function.
    """
    batch_size, n_traces, n_times = data.shape
    data_fx = fourier_transform_time(data, time_window_len)
    data_fx_freq_windowed = extract_freq_window(data_fx, min_freq, max_freq)
    # count will store the number of estimates of each component that have been
    # summed, so dividing the sum of estimates by count gives the mean
    count = torch.zeros_like(data_fx_freq_windowed[..., 0])

    trace_window_starts = torch.arange(
        0, n_traces - trace_window_len // 2, trace_window_len // 2
    )
    data_fx_trace_windowed, final_trace_window_len = extract_trace_windows(
        data_fx_freq_windowed, trace_window_starts, trace_window_len
    )

    # data_fx_trace_windowed is of shape
    # [batch_size, n_freqs, n_time_windows,
    # n_trace_windows, trace_window_len, 2]. We will calculate the magnitude
    # (complex norm) and then calculate the median of that across
    # trace windows.
    magnitude = complex_norm(data_fx_trace_windowed).unsqueeze(-1)
    median, _ = magnitude.median(dim=-2, keepdim=True)
    magnitude = magnitude.expand_as(data_fx_trace_windowed)
    median = median.expand_as(data_fx_trace_windowed)
    should_clip = magnitude > clip_factor * median
    data_fx_trace_windowed[should_clip] *= (median[should_clip]
                                            / magnitude[should_clip])

    data_fx_denoised, count =  combine_trace_windows(
        data_fx_trace_windowed,
        trace_window_starts,
        trace_window_len,
        0,
        data_fx.shape,
    )

    # Divide by count
    # A small value is added to the count to prevent the possibility
    # of a division by zero
    data_fx_denoised /= count[..., None] + 1e-10

    # Restore frequencies outside the range the tool was applied to
    data_fx_denoised = restore_freq_window(
        data_fx, data_fx_denoised, min_freq, max_freq
    )

    # Inverse Fourier transform in time
    return inverse_fourier_transform_time(data_fx_denoised, time_window_len,
                                           n_times)
