import torch
import numpy as np
import pytest
from pytest import approx
from seispro import agc

def test_constant(batch_size, n_traces, n_times, time_window_side_len):
    if 2*time_window_side_len+1 > n_times:
        return
    data = torch.randn(batch_size, n_traces, 1).repeat(1, 1, n_times)
    expected = torch.sign(data) * torch.ones_like(data)
    data_out, _ = agc(data, time_window_side_len)
    mse = ((expected - data_out)**2).mean()
    assert mse == approx(0.0, abs=1e-6)


# Removing this test as things are a bit more complicated than I first
# imagined - I forgot that if the signal changes sign then this won't
# pass. I could probably enforce that the slope is chosen so that it
# doesn't change sign, but that seems overkill.
#def test_linear(batch_size, n_traces, n_times, time_window_side_len):
#    if 2*time_window_side_len+1 > n_times:
#        return
#    data_intercept = torch.randn(batch_size, n_traces, 1)
#    data_slope = torch.randn(batch_size, n_traces, 1)
#    data = (data_intercept
#            + torch.arange(1, n_times + 1).reshape(1, 1, -1)
#            * data_slope)
#    expected = torch.sign(data) * torch.ones_like(data)
#    data_out, _ = agc(data, time_window_side_len)
#    # The edges are not expected to be 1 (due to edge effects), so ignore them
#    mse = ((expected - data_out)[...,time_window_side_len:-time_window_side_len]**2).mean()
#    assert mse == approx(0.0, abs=1e-6)


def test_reverse(batch_size, n_traces, n_times, time_window_side_len):
    if 2*time_window_side_len+1 > n_times:
        return
    data = torch.randn(batch_size, n_traces, n_times)
    data_out, scaling = agc(data, time_window_side_len)
    data_agc_reversed = data_out / (scaling + 1e-10)
    mse = ((data - data_agc_reversed)**2).mean()
    assert mse == approx(0.0, abs=1e-6)


def pytest_generate_tests(metafunc):
    if 'batch_size' in metafunc.fixturenames:
        metafunc.parametrize('batch_size', [1, 3, 4])
    if 'n_traces' in metafunc.fixturenames:
        metafunc.parametrize('n_traces', [1, 63, 64])
    if 'n_times' in metafunc.fixturenames:
        metafunc.parametrize('n_times', [1, 63, 64])
    if 'time_window_side_len' in metafunc.fixturenames:
        metafunc.parametrize('time_window_side_len', [1, 15, 16])
