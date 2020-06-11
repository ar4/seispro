"""
2D FXDECON
"""
import torch
import torchaudio
import numpy as np


def _toeplitz(data_fx, filter_len):
    """Constructs a Toeplitz matrix corresponding to application of filter.

    Inputs:
        data_fx: [batch_size, trace_window_len, 2]
        filter_len: An integer specifying the length of the filter

    Returns:
        X = [x_0, x_1, ..., x_{filter_len-1}]
            [x_1, x_2, ..., x_{filter_len}]
            ...
            [x_{trace_window_len-filter_len-1}, ..., x_{trace_window_len-2}]
        X: [-1, trace_window_len-filter_len, filter_len, 2]
    """
    # type: (Tensor, int) -> Tensor
    batch_size, trace_window_len, _ = data_fx.shape
    # :-1 as the final trace in each trace window is not an input to any filter
    # The output of unfold is permuted as unfold outputs
    # [batch_size, trace_window_len-filter_len, 2, filter_len]
    # The second dimension has size trace_window_len-filter_len
    # because unfold on a dimension of size a with a filter size b
    # would return an output of size a - (b-1). Since in our case
    # a = trace_window_len - 1 (-1 due to -1 in data_fx[:, :-1] below)
    # b = filter_len, so output size is trace_window_len - 1 - (filter_len - 1)
    X = data_fx[:, :-1].unfold(1, filter_len, 1).permute(0, 1, 3, 2)
    assert X.shape == (
        batch_size,
        trace_window_len - filter_len,
        filter_len,
        2,
    ), "X {}, expected {}".format(
        X.shape, (batch_size, trace_window_len - filter_len, filter_len, 2)
    )
    return X


def _wiener_denoise(data_fx, filter_len, final_trace_window_len):
    """Uses a prediction filter to estimate signal predictable from the left.

    data_fx: [batch_size, n_freqs, n_time_windows, n_trace_windows,
              trace_window_len, 2]
    """
    # type: (Tensor, int, int) -> Tensor
    (
        batch_size,
        n_freqs,
        n_time_windows,
        n_trace_windows,
        trace_window_len,
        _,
    ) = data_fx.shape
    dtype = data_fx.dtype
    device = data_fx.device
    # Treat each frequency of each trace window independently
    data_fx = data_fx.reshape(-1, trace_window_len, 2)
    X = _toeplitz(data_fx, filter_len).clone()
    # X: [batch_size * n_freqs * n_time_windows * n_trace_windows,
    #     trace_window_len - filter_len, filter_len, 2]
    # The final trace_window_len - final_trace_window_len elements of the
    # last window are zero, so set those rows of X to zero as we do not wish
    # to use them when constructing the prediction filter (using non-zero values
    # to try to predict zero values will bias the filter).
    # I would like to do the following, but there is currently a bug that prevents
    # this from working (https://github.com/pytorch/pytorch/issues/38555)
    #X.reshape(
    #    batch_size, n_freqs, n_time_windows, n_trace_windows, X.shape[-3], filter_len, 2
    #)[:, :, :, -1, X.shape[-3] - (trace_window_len - final_trace_window_len):].fill_(0.0)
    # So instead reshape, zero the rows, and then undo the reshape
    Xshape = X.shape
    X = X.reshape(
        batch_size * n_freqs * n_time_windows, n_trace_windows, X.shape[-3], filter_len, 2
    )
    X[:, -1, X.shape[-3] - (trace_window_len - final_trace_window_len) :, :, :] = 0.0
    X = X.reshape(Xshape[0], Xshape[1], Xshape[2], Xshape[3])

    # R = X^HX
    # R: [..., filter_len, filter_len, 2]
    # P = Re(R) = Re(X^H)*Re(X) - Im(X^H)Im(X)
    # Q = Im(R) = Re(X^H)*Im(X) + Im(X^H)Re(X)
    batch_size2 = X.shape[0]  # batch dim is everything except window, re/im
    XH = torch.transpose(X.clone(), -3, -2)
    XH[..., 1] *= -1
    P = XH[..., 0] @ X[..., 0] - XH[..., 1] @ X[..., 1]
    Q = XH[..., 0] @ X[..., 1] + XH[..., 1] @ X[..., 0]
    assert P.shape == (batch_size2, filter_len, filter_len), "P {}, expected {}".format(
        P.shape, (batch_size2, filter_len, filter_len)
    )
    M = torch.zeros(
        batch_size2, 2 * filter_len, 2 * filter_len, device=device, dtype=dtype
    )
    # M: [P, -Q]
    #    [Q, P]
    M[:, :filter_len, :filter_len] = P
    M[:, :filter_len, filter_len:] = -Q
    M[:, filter_len:, :filter_len] = Q
    M[:, filter_len:, filter_len:] = P
    # g: X^H*d, where d is desired output
    # desired_output: [..., trace_window_len-filter_len]
    # g: [..., filter_len, 1]
    # Need to add length 1 dim to end of desired output to get matmul
    desired_output = data_fx[:, filter_len:]
    g_re = (
        XH[..., 0] @ desired_output[..., 0, None]
        - XH[..., 1] @ desired_output[..., 1, None]
    )
    g_im = (
        XH[..., 1] @ desired_output[..., 0, None]
        + XH[..., 0] @ desired_output[..., 1, None]
    )
    # A small value is added to the diagonal of M for stability
    filt, _ = torch.solve(
        torch.cat([g_re, g_im], dim=1),
        M + 1e-5 * torch.eye(2 * filter_len, device=device, dtype=dtype)[None],
    )
    assert filt.shape == (batch_size2, 2 * filter_len, 1), "filt {}, expected {}".format(
        filt.shape, (batch_size2, filter_len, 1)
    )
    output = torch.zeros_like(desired_output)
    filt_re = filt[:, :filter_len]
    filt_im = filt[:, filter_len:]
    # Now use filter to create output, y = Xf
    output[..., 0] = (X[..., 0] @ filt_re - X[..., 1] @ filt_im)[..., 0]
    output[..., 1] = (X[..., 0] @ filt_im + X[..., 1] @ filt_re)[..., 0]
    # output: [batch_size, n_freqs, n_time_windows, n_trace_windows, trace_window_len-filter_len, 2]
    return output.reshape(
        batch_size,
        n_freqs,
        n_time_windows,
        n_trace_windows,
        trace_window_len - filter_len,
        2,
    )


def _combine_trace_windows(
    data_fx_windowed, trace_window_starts, trace_window_len, filter_len, data_fx_shape
):
    """Combines trace windows.

    Input data_fx_windowed: [batch_size, n_freqs, n_time_windows,
                             n_trace_windows, trace_window_len-filter_len, 2]
    Return data_fx: [batch_size, n_freqs, n_time_windows, n_traces, 2]
    """
    # type: (Tensor, Tensor, int, int, List[int]) -> Tuple[Tensor, Tensor]
    dtype = data_fx_windowed.dtype
    device = data_fx_windowed.device
    data_fx_combined = torch.zeros(data_fx_shape[0], data_fx_shape[1], data_fx_shape[2], data_fx_shape[3], data_fx_shape[4], device=device, dtype=dtype)
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


def _extract_trace_windows(data_fx, trace_window_starts, trace_window_len):
    """Extracts windows of traces from the data.

    Returns:
        data_fx_trace_windowed: A Tensor of shape
                                [batch_size, n_freqs, n_time_windows,
                                 n_trace_windows, trace_window_len, 2]
        final_trace_window_len: An integer specifying the number of traces
                                in the final trace window
    """
    # type: (Tensor, Tensor, int) -> Tuple[Tensor, int]
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


def _fxdecon_one_direction(data_fx, filter_len, trace_window_len):
    """Applies FXDECON in one direction.

    This will create filters on windows of traces that use a
    group of traces to predict the trace to the right of the group.
    The predicted traces are the output.

    Traces are only predicted when a full filter of traces are
    available for the prediction, so the first predicted trace
    is at index filter_len + 1.
    """
    # type: (Tensor, int, int) -> Tuple[Tensor, Tensor]
    batch_size, n_freqs, n_time_windows, n_traces, _ = data_fx.shape
    trace_window_starts = torch.arange(
        0, n_traces - trace_window_len // 2, trace_window_len // 2
    )
    data_fx_trace_windowed, final_trace_window_len = _extract_trace_windows(
        data_fx, trace_window_starts, trace_window_len
    )

    data_fx_trace_windowed_denoised = _wiener_denoise(
        data_fx_trace_windowed, filter_len, final_trace_window_len
    )
    # data_fx_trace_windowed_denoised: [batch_size, n_freqs, n_time_windows,
    #                                   n_trace_windows,
    #                                   trace_window_len-filter_len, 2]

    return _combine_trace_windows(
        data_fx_trace_windowed_denoised,
        trace_window_starts,
        trace_window_len,
        filter_len,
        data_fx.shape,
    )


def _restore_freq_window(data_fx, data_fx_denoised, min_freq, max_freq):
    """Replaces components between min and max freq with denoised values.

    Inputs:
        data_fx: [batch_size, n_freqs, n_time_windows, n_traces, 2]
        data_fx_denoised: [batch_size, n_freqs*(max_freq-min_freq), n_time_windows, n_traces, 2]
        min_freq, max_freq: Floats specifying min and max Nyquist fractions

    Returns:
        data_fx: [batch_size, n_freqs, n_time_windows, n_traces, 2] after the
                 values between min and max freq have been replaced by those
                 in data_fx_denoised.
    """
    # type: (Tensor, Tensor, float, float) -> Tensor
    n_freqs = data_fx.shape[1]
    min_freq_idx = int(n_freqs * min_freq)
    max_freq_idx = int(n_freqs * max_freq)
    data_fx[:, min_freq_idx:max_freq_idx] = data_fx_denoised
    return data_fx


def _extract_freq_window(data_fx, min_freq, max_freq):
    """Extracts components corresponding to frequencies between min and max.

    Inputs:
        data_fx: [batch_size, n_freqs, n_time_windows, n_traces, 2]
        min_freq, max_freq: Floats specifying min and max Nyquist fractions

    Returns:
        data_fx_freq_windowed: [batch_size, n_freqs*(max_freq-min_freq), n_time_windows, n_traces, 2]
    """
    # type: (Tensor, float, float) -> Tensor
    n_freqs = data_fx.shape[1]
    min_freq_idx = int(n_freqs * min_freq)
    max_freq_idx = int(n_freqs * max_freq)
    return data_fx[:, min_freq_idx:max_freq_idx]


def _inverse_fourier_transform_time(data_fx, time_window_len, n_times):
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
    # type: (Tensor, int, int) -> Tensor
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
    data = torchaudio.functional.istft(data_fx, time_window_len, hop_length=time_window_len//2,
                                       window=time_window, length=n_times)
    return data.reshape(batch_size, n_traces, n_times)


def _fourier_transform_time(data, time_window_len):
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
    # type: (Tensor, int) -> Tensor
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
    data_fx = torch.stft(data.reshape(-1, n_times), time_window_len,
                         hop_length=time_window_len//2, window=time_window)
    n_freqs, n_time_windows = data_fx.shape[1:3]
    data_fx = data_fx.reshape(batch_size, n_traces, n_freqs, n_time_windows, 2)
    data_fx = data_fx.permute(0, 2, 3, 1, 4)
    return data_fx


def fxdecon(
    data,
    filter_len=4,
    trace_window_len=12,
    time_window_len=512,
    min_freq=0.0,
    max_freq=1.0,
):
    """Applies 2D FXDECON to attenuate random noise.

    This process uses neighbouring traces to predict the frequency componen_timess
    of each trace. In this way, it attenuates unpredictable features in the
    data, which are assumed to be noise. The input data is windowed in the
    time and trace dimensions before the prediction filters are created.

    Although FXDECON is often quite effective at removing noise, it also tends
    to remove signal.

    Inputs:
        data: A [batch_size, n_traces, n_times] shape Tensor containing the data
        filter_len: An integer specifying the length of the prediction
                    filter. It should normally be in the range 3-11.
                    Using a larger filter reduces noise attenuation
                    but increases data fidelity. Default 4.
        trace_window_len: An integer specifying the window length in the trace
                          dimension to use when calculating the autocorrelation.
                          It should normally be 3-4 times the value of
                          filter_len, and must be at least 2 times.
                          Default 12.
        time_window_len: An integer specifying the window length in the time
                         dimension to use when Fourier transforming the data.
                         Default 512.
        min_freq: A float specifying the minimum fraction of the
                  Nyquist frequency to apply FXDECON to. Default: 0.
        max_freq: A float specifying the maximum fraction of the
                  Nyquist frequency to apply FXDECON to. Default: 1.

    Returns:
        data: A Tensor of the same shape as the input, after filtering.
    """
    # type: (Tensor, int, int, int, float, float) -> Tensor
    batch_size, n_traces, n_times = data.shape
    if n_traces < 2:
        raise RuntimeError('number of traces must be >= 2')
    if not np.issubdtype(type(filter_len), np.integer):
        raise RuntimeError('filter_len must be an integer')
    if not 1 <= filter_len < trace_window_len:
        raise RuntimeError('filter_len must be in [1, trace_window_len//2]')
    if not np.issubdtype(type(trace_window_len), np.integer):
        raise RuntimeError('trace_window_len must be an integer')
    if not 2 * filter_len <= trace_window_len <= n_traces:
        raise RuntimeError('trace_window_len must be in [2*filter_len, n_traces]')
    if not np.issubdtype(type(time_window_len), np.integer):
        raise RuntimeError('time_window_len must be an integer')
    if not 2 <= time_window_len <= n_times:
        raise RuntimeError('time_window_len must be in [2, n_times]')
    if not 0.0 <= min_freq < 1.0:
        raise RuntimeError('min_freq must be in [0.0, 1.0)')
    if not 0.0 < max_freq <= 1.0:
        raise RuntimeError('max_freq must be in (0.0, 1.0]')
    return fxdecon_jit(data, filter_len, trace_window_len, time_window_len, min_freq, max_freq)


@torch.jit.script
def fxdecon_jit(
    data,
    filter_len,
    trace_window_len,
    time_window_len,
    min_freq,
    max_freq,
):
    """JIT-compiled function without error checking.

    raise seems to not be supported by Torchscript, so error checking
    is done before calling this JIT-compiled function.
    """
    # type: (Tensor, int, int, int, float, float) -> Tensor
    batch_size, n_traces, n_times = data.shape
    data_fx = _fourier_transform_time(data, time_window_len)
    data_fx_freq_windowed = _extract_freq_window(data_fx, min_freq, max_freq)
    data_fx_denoised = torch.zeros_like(data_fx_freq_windowed)
    # count will store the number of estimates of each componen_times that have been
    # summed, so dividing the sum of estimates by count gives the mean
    count = torch.zeros_like(data_fx_freq_windowed[..., 0])

    # FXDECON predicting one cell to the right
    data_fx_dir, count_dir = _fxdecon_one_direction(
        data_fx_freq_windowed, filter_len, trace_window_len
    )
    data_fx_denoised += data_fx_dir
    count += count_dir

    # FXDECON predicting one cell to the left
    # flip the input in the trace dimension (-2)
    data_fx_dir, count_dir = _fxdecon_one_direction(
        data_fx_freq_windowed.flip(-2), filter_len, trace_window_len,
    )
    # flip the output back before adding to previous estimates
    data_fx_denoised += data_fx_dir.flip(-2)
    count += count_dir.flip(-1)

    # Divide by count
    # A small value is added to the count to prevent the possibility
    # of a division by zero
    data_fx_denoised /= count[..., None] + 1e-10

    # Restore frequencies outside the range FXDECON was applied to
    data_fx_denoised = _restore_freq_window(
        data_fx, data_fx_denoised, min_freq, max_freq
    )

    # Inverse Fourier transform in time
    return _inverse_fourier_transform_time(data_fx_denoised, time_window_len, n_times)
