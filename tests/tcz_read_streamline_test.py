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
    # TCZ having a streamline with all its points
    tcz = Tcz('tests/dicelib/mock/demo_fibers_to_smooth.tcz', mode='r')
    n_points, streamline = tcz.read_streamline()
    assert n_points == 100

    assert streamline[0][0] == pytest.approx(49.40, abs=0.01)
    assert streamline[0][1] == pytest.approx(-1.98, abs=0.01)
    assert streamline[0][2] == pytest.approx(22.95, abs=0.01)

    assert streamline[1][0] == pytest.approx(24.5, abs=0.01)
    assert streamline[1][1] == pytest.approx(48.9, abs=0.01)
    assert streamline[1][2] == pytest.approx(-1.98, abs=0.01)

    assert streamline[2][0] == pytest.approx(22.95, abs=0.01)
    assert streamline[2][1] == pytest.approx(24.5, abs=0.01)
    assert streamline[2][2] == pytest.approx(48.31, abs=0.01)

    assert streamline[3][0] == pytest.approx(-1.98, abs=0.01)
    assert streamline[3][1] == pytest.approx(22.95, abs=0.01)
    assert streamline[3][2] == pytest.approx(24.5, abs=0.01)

    assert streamline[4][0] == pytest.approx(-1.98, abs=0.01)
    assert streamline[4][1] == pytest.approx(45.9, abs=0.01)
    assert streamline[4][2] == pytest.approx(-1.98, abs=0.01)

    assert streamline[5][0] == pytest.approx(22.95, abs=0.01)
    assert streamline[5][1] == pytest.approx(24.5, abs=0.01)
    assert streamline[5][2] == pytest.approx(-1.98, abs=0.01)

    assert streamline[6][0] == pytest.approx(45.9, abs=0.01)
    assert streamline[6][1] == pytest.approx(-1.98, abs=0.01)
    assert streamline[6][2] == pytest.approx(22.95, abs=0.01)

    assert streamline[7][0] == pytest.approx(24.5, abs=0.01)
    assert streamline[7][1] == pytest.approx(-1.98, abs=0.01)
    assert streamline[7][2] == pytest.approx(45.9, abs=0.01)

    assert streamline[8][0] == pytest.approx(-1.98, abs=0.01)
    assert streamline[8][1] == pytest.approx(22.95, abs=0.01)
    assert streamline[8][2] == pytest.approx(24.5, abs=0.01)

    assert streamline[9][0] == pytest.approx(-1.98, abs=0.01)
    assert streamline[9][1] == pytest.approx(45.9, abs=0.01)
    assert streamline[9][2] == pytest.approx(-1.98, abs=0.01)

    assert streamline[10][0] == pytest.approx(22.95, abs=0.01)
    assert streamline[10][1] == pytest.approx(24.5, abs=0.01)
    assert streamline[10][2] == pytest.approx(-1.98, abs=0.01)

    assert streamline[11][0] == pytest.approx(45.9, abs=0.01)
    assert streamline[11][1] == pytest.approx(-1.98, abs=0.01)
    assert streamline[11][2] == pytest.approx(22.95, abs=0.01)

    assert streamline[12][0] == pytest.approx(24.5, abs=0.01)
    assert streamline[12][1] == pytest.approx(-1.98, abs=0.01)
    assert streamline[12][2] == pytest.approx(45.9, abs=0.01)

    assert streamline[13][0] == pytest.approx(-1.98, abs=0.01)
    assert streamline[13][1] == pytest.approx(22.95, abs=0.01)
    assert streamline[13][2] == pytest.approx(24.5, abs=0.01)

    assert streamline[14][0] == pytest.approx(-1.98, abs=0.01)
    assert streamline[14][1] == pytest.approx(45.9, abs=0.01)
    assert streamline[14][2] == pytest.approx(-1.98, abs=0.01)

    assert streamline[15][0] == pytest.approx(22.95, abs=0.01)
    assert streamline[15][1] == pytest.approx(24.5, abs=0.01)
    assert streamline[15][2] == pytest.approx(-1.98, abs=0.01)

    tcz.close()
