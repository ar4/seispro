import torch
import numpy as np
import pytest
from pytest import approx
from seispro import fxdecon

def run_fxdecon(data, filter_len, trace_window_len, time_window_len):
    batch_size, n_traces, n_times = data.shape
    if filter_len > n_traces or not (2*filter_len <= trace_window_len <= n_traces) or time_window_len > n_times:
        with pytest.raises(RuntimeError):
            data_out = fxdecon.fxdecon(data, filter_len, trace_window_len,
                                       time_window_len)
    else:
        data_out = fxdecon.fxdecon(data, filter_len, trace_window_len,
                                   time_window_len)
        mse = ((data - data_out)**2).mean()
        assert mse == approx(0.0, abs=1e-6)


def test_constant_horizontal(batch_size, n_traces, n_times, filter_len,
                             trace_window_len, time_window_len):
    data = torch.randn(batch_size, 1, n_times).repeat(1, n_traces, 1)
    run_fxdecon(data, filter_len, trace_window_len, time_window_len)


def test_linear_horizontal(batch_size, n_traces, n_times, filter_len,
                           trace_window_len, time_window_len):
    if filter_len == 1:
        return
    data = (torch.randn(batch_size, 1, n_times)
            + torch.arange(1, n_traces + 1).reshape(1, -1, 1)
            * torch.randn(batch_size, 1, n_times))
    run_fxdecon(data, filter_len, trace_window_len, time_window_len)


def test_constant_dipping(batch_size, n_traces, n_times, filter_len,
                          trace_window_len, time_window_len):
    if filter_len == 1:
        return
    data = torch.zeros(batch_size, n_traces, n_times)
    if n_times > n_traces:
        dipping_data = torch.randn(batch_size, n_times-n_traces)
        for traceidx in range(n_traces):
            data[:, traceidx, traceidx:traceidx + n_times-n_traces] = \
                dipping_data
    run_fxdecon(data, filter_len, trace_window_len, time_window_len)


def test_linear_dipping(batch_size, n_traces, n_times, filter_len,
                        trace_window_len, time_window_len):
    if filter_len == 1:
        return
    data = torch.zeros(batch_size, n_traces, n_times)
    if n_times > n_traces:
        dipping_data = torch.randn(batch_size, n_times-n_traces)
        linear_slope = torch.randn(batch_size, n_times-n_traces)
        for traceidx in range(n_traces):
            data[:, traceidx, traceidx:traceidx + n_times-n_traces] = \
                dipping_data + traceidx * linear_slope
    run_fxdecon(data, filter_len, trace_window_len, time_window_len)


def pytest_generate_tests(metafunc):
    if 'batch_size' in metafunc.fixturenames:
        metafunc.parametrize('batch_size', [1, 3, 4])
    if 'n_traces' in metafunc.fixturenames:
        metafunc.parametrize('n_traces', [2, 3, 4])
    if 'n_times' in metafunc.fixturenames:
        metafunc.parametrize('n_times', [2, 3, 4])
    if 'filter_len' in metafunc.fixturenames:
        metafunc.parametrize('filter_len', np.arange(1, 4))
    if 'trace_window_len' in metafunc.fixturenames:
        metafunc.parametrize('trace_window_len', np.arange(2, 4))
    if 'time_window_len' in metafunc.fixturenames:
        metafunc.parametrize('time_window_len', np.arange(2, 4))
