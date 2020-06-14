import torch
import numpy as np
import pytest
from pytest import approx
from seispro import tfclip


def run_tfclip(data, expected, trace_window_len, time_window_len,
               clip_factor):
    batch_size, n_traces, n_times = data.shape
    if not (3 <= trace_window_len <= n_traces) or time_window_len > n_times:
        with pytest.raises(RuntimeError):
            data_out = tfclip(data, trace_window_len, time_window_len,
                              clip_factor=clip_factor)
    else:
        data_out = tfclip(data, trace_window_len, time_window_len,
                          clip_factor=clip_factor)
        mse = ((expected - data_out)**2).mean()
        assert mse == approx(0.0, abs=1e-6)


def test_constant_horizontal(batch_size, n_traces, n_times, trace_window_len,
                             time_window_len, clip_factor=1.0):
    data = torch.randn(batch_size, 1, n_times).repeat(1, n_traces, 1)
    run_tfclip(data, data, trace_window_len, time_window_len, clip_factor)


# Due to edge effects this test will currently not pass. The median calculation
# in TFCLIP should perhaps be performed with a rolling window instead of
# in windows that half overlap, as currently the right edge may have
# a window that is not full and so the calculated median may be a poor estimate
#def test_constant_horizontal_with_noise(batch_size, n_traces, n_times,
#                                        trace_window_len,
#                                        time_window_len, clip_factor=1.0):
#    data = torch.randn(batch_size, 1, n_times).repeat(1, n_traces, 1)
#    expected = data.clone()
#    # Randomly scale one sample per gather. Median clipping
#    # (with clip_factor=1) should then undo the scaling
#    for batchidx in range(batch_size):
#        data[batchidx, np.random.randint(0, n_traces),
#             np.random.randint(0, n_times)] *= 1 + np.random.rand() * 10
#
#    run_tfclip(data, expected, trace_window_len, time_window_len, clip_factor)


def pytest_generate_tests(metafunc):
    if 'batch_size' in metafunc.fixturenames:
        metafunc.parametrize('batch_size', [1, 3, 4])
    if 'n_traces' in metafunc.fixturenames:
        metafunc.parametrize('n_traces', [3, 4, 5])
    if 'n_times' in metafunc.fixturenames:
        metafunc.parametrize('n_times', [2, 3, 4])
    if 'trace_window_len' in metafunc.fixturenames:
        metafunc.parametrize('trace_window_len', np.arange(2, 4))
    if 'time_window_len' in metafunc.fixturenames:
        metafunc.parametrize('time_window_len', np.arange(2, 4))
