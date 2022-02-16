from decimal import Decimal
from fractions import Fraction

import gspaces
import numpy as np
import pytest


@pytest.mark.parametrize(
    "start,stop,step",
    [
        (0, 100, 10),
        (0.5, 10.5, 4),
        pytest.param(0, 10, -1, marks=pytest.mark.xfail),
        (Fraction(1, 2), Fraction(36, 4), Fraction(1, 3)),
        # (Decimal(1) / Decimal(2), Decimal(36) / Decimal(4), Decimal(1) / Decimal(3)),
    ],
)
def test_arange(start, stop, step):
    _numpy = np.arange(start, stop, step)
    _gspaces = gspaces.arange(start, stop, step)

    assert all(x == y for x, y in zip(_numpy, _gspaces))


@pytest.mark.parametrize(
    "start,stop,num",
    [
        (0, 10, 100),
        (0.5, 10.5, 30),
        pytest.param(0, 10, -1, marks=pytest.mark.xfail),
        # (Fraction(1, 2), Fraction(36, 4), 30),
        # (Decimal(1) / Decimal(2), Decimal(36) / Decimal(4), Decimal(1) / Decimal(3)),
    ],
)
def test_linspace(start, stop, num):
    _numpy = np.linspace(start, stop, num)
    _gspaces = gspaces.linspace(start, stop, num)

    assert all(x == pytest.approx(y) for x, y in zip(_numpy, _gspaces))
