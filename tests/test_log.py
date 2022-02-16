import gspaces
import numpy as np
import pytest


@pytest.mark.parametrize(
    "start,stop,num,endpoint,base",
    [
        (0, 100, 10, True, 10.0),
        (0.5, 10.5, 4, True, 10.0),
        pytest.param(0, 10, -1, True, 10.0, marks=pytest.mark.xfail),
    ],
)
def test_logspace(start, stop, num, endpoint, base):
    _numpy = np.logspace(start, stop, num=num, endpoint=endpoint, base=base)
    _gspaces = gspaces.logspace(start, stop, num=num, endpoint=endpoint, base=base)

    assert all(x == pytest.approx(y) for x, y in zip(_numpy, _gspaces))


@pytest.mark.parametrize(
    "start,stop,num,endpoint",
    [
        pytest.param(0, 100, 10, True, marks=pytest.mark.xfail),
        (0.5, 10.5, 4, True),
        pytest.param(0, 10, -1, True, marks=pytest.mark.xfail),
    ],
)
def test_geomspace(start, stop, num, endpoint):
    _numpy = np.geomspace(start, stop, num=num, endpoint=endpoint)
    _gspaces = gspaces.geomspace(start, stop, num=num, endpoint=endpoint)

    assert all(x == pytest.approx(y) for x, y in zip(_numpy, _gspaces))
