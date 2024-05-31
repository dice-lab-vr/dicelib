from dicelib.tcz import Tcz
import pytest


def test_read_fake_streamline_successfully():
    tcz = Tcz('tests/dicelib/mock/demo_fibers_read_streamline.tcz', 'r')
    n_points, streamline = tcz.read_streamline()
    for x in range(n_points):
        for y in range(3):
            assert streamline[x][y] == 132.375
    tcz.close()


def test_read_streamline_with_spline_with_little_epsilon_will_return_all_points():

    tcz = Tcz('tests/dicelib/mock/demo_fibers_to_smooth.tcz', mode='r')
    n_points, streamline = tcz.read_streamline()
    assert n_points == 100

    # check only some points
    assert streamline[0][0] == 49.40625
    assert streamline[0][1] == 22.0
    assert streamline[0][2] == 24.5

    assert streamline[1][0] == 48.90625
    assert streamline[1][1] == 22.046875
    assert streamline[1][2] == 24.5

    assert streamline[2][0] == 48.40625
    assert streamline[2][1] == 22.09375
    assert streamline[2][2] == 24.5

    assert streamline[3][0] == 47.90625
    assert streamline[3][1] == 22.140625
    assert streamline[3][2] == 24.5

    assert streamline[4][0] == 47.40625
    assert streamline[4][1] == 22.171875
    assert streamline[4][2] == 24.5

    assert streamline[5][0] == 46.90625
    assert streamline[5][1] == 22.203125
    assert streamline[5][2] == 24.5

    assert streamline[6][0] == 46.40625
    assert streamline[6][1] == 22.234375
    assert streamline[6][2] == 24.5

    assert streamline[7][0] == 45.90625
    assert streamline[7][1] == 22.265625
    assert streamline[7][2] == 24.5

    assert streamline[8][0] == 45.375
    assert streamline[8][1] == 22.28125
    assert streamline[8][2] == 24.5

    assert streamline[9][0] == 44.875
    assert streamline[9][1] == 22.296875
    assert streamline[9][2] == 24.5

    assert streamline[10][0] == 44.375
    assert streamline[10][1] == 22.3125
    assert streamline[10][2] == 24.5

    assert streamline[11][0] == 43.875
    assert streamline[11][1] == 22.328125
    assert streamline[11][2] == 24.5

    assert streamline[12][0] == 43.375
    assert streamline[12][1] == 22.34375
    assert streamline[12][2] == 24.5

    assert streamline[13][0] == 42.875
    assert streamline[13][1] == 22.359375
    assert streamline[13][2] == 24.5

    assert streamline[14][0] == 42.375
    assert streamline[14][1] == 22.359375
    assert streamline[14][2] == 24.5

    assert streamline[15][0] == 41.875
    assert streamline[15][1] == 22.359375
    assert streamline[15][2] == 24.5

    assert streamline[99][0] == -0.39990234375
    assert streamline[99][1] == 22.0
    assert streamline[99][2] == 24.5

    tcz.close()
