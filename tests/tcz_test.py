import numpy as np
import pytest  # pip install pytest # build/temp.linux-x86_64-cpython-311/build/dicelib/tcz.o

from dicelib.tcz import Tcz


# COMPILE and INSTALL the library
# Building is required, as we compile python to C code
# first build
# $> python setup.py build && python setup.py build_ext --inplace && python setup.py install && pip install . -U
# to spare time, if only extensions are modified, the following instruction is the only one required for building
# $> python setup.py build_ext
# RUN TESTS
# $> pytest

def test_create_in_read_mode_successfully():
    tcz = Tcz('tests/dicelib/mock/demo_fibers.tcz', 'r', None, 1000)
    assert tcz.header['blur_core_extent'] == 12.3
    assert tcz.header['blur_gauss_extent'] == 34.4
    assert tcz.header['blur_spacing'] == 23.1
    assert tcz.header['blur_gauss_min'] == 34.0
    assert tcz.header['representation'] == 'polyline'
    assert tcz.header['datatype'] == 'Float16'
    assert tcz.header['segment_len'] == 0.8
    assert tcz.header['epsilon'] == 0.4
    assert tcz.max_points == 1000
    assert len(tcz.streamline) == 1000
    assert type(tcz.streamline[0][0]) is float


def test_create_in_read_mode_no_segment_len_will_setup_segment_len_automatically():
    tcz = Tcz('tests/dicelib/mock/demo_fibers_no_segment_len.tcz', 'r', None, 1000)
    assert tcz.header['segment_len'] == 0.5


def test_create_in_read_mode_streamline_spline_successfully():
    tcz = Tcz('tests/dicelib/mock/demo_fibers_streamline_spline.tcz', 'r', None, 1000)
    assert tcz.header['representation'] == 'spline'


def test_create_in_write_mode_successfully():
    header_test = {
        'blur_core_extent': '1.1',
        'blur_gauss_extent': '2.2',
        'blur_spacing': '3.3',
        'blur_gauss_min': '4.4',
        'representation': 'polyline',
        'datatype': 'Float32LE',
        'count': '999',
        'timestamp': '2040-01-01T00:00:00.000Z',
    }
    tcz = Tcz('tests/dicelib/mock/demo_fibers_write.tcz', 'w', header_test)
    assert tcz.streamline is None


def test_create_in_read_mode_no_representation_in_file_will_fall_back_to_polyline():
    tcz = Tcz('tests/dicelib/mock/demo_fibers_no_polyline.tcz', 'r', )
    assert tcz.header['representation'] == 'polyline'
    assert tcz.header['datatype'] == 'Float16'


def test_write_streamline_successfully():
    header_test = {
        'blur_core_extent': '1.1',
        'blur_gauss_extent': '2.2',
        'blur_spacing': '3.3',
        'blur_gauss_min': '4.4',
        'epsilon': '0.4',
        'segment_len': '0.5',
        'representation': 'polyline',
        'datatype': 'Float32LE',
        'count': '999',
        'timestamp': '2040-01-01T00:00:00.000Z',
    }
    tcz = Tcz('tests/dicelib/mock/demo_fibers_write_streamline.tcz', 'w', header_test)
    tcz.n_pts = 4
    fake_streamline = np.full((tcz.n_pts, 3), fill_value=132.364, dtype=np.float32)
    tcz.write_streamline(fake_streamline, tcz.n_pts)


def test_write_streamline_spline_will_smooth_streamline():
    header_test = {
        'blur_core_extent': '1.1',
        'blur_gauss_extent': '2.2',
        'blur_spacing': '3.3',
        'blur_gauss_min': '4.4',
        'epsilon': '2.2',
        'segment_len': '0.5',
        'representation': 'spline',
        'datatype': 'Float32LE',
        'count': '999',
        'timestamp': '2040-01-01T00:00:00.000Z',
        'total_count': '1'
    }
    tcz_out = Tcz('tests/dicelib/mock/demo_fibers_smoothed.tcz', mode='w', header=header_test)
    tcz_out.n_pts = 6
    fake_streamline = np.array([
        [2.1, 3.2, 1.3],
        [1.1, 2.2, 3.3],
        [1.0, 0.5, 5.4],
        [1.9, 8.1, 1.1],
        [0.8, 0.2, 0.1],
        [6.8, 4.2, 5.1],
    ], dtype=np.float32)
    tcz_out.write_streamline(fake_streamline, 6)
    tcz_out.close(False)


def test_read_streamline_with_spline_with_little_epsilon_will_return_all_points():
    # TCZ having a streamline with all its points
    tcz_out = Tcz('tests/dicelib/mock/demo_fibers_to_smooth.tcz', mode='r')
    n_points = tcz_out.read_streamline()
    assert n_points == 100

    # checking the first four points only
    assert tcz_out.streamline[0][0] == pytest.approx(49.40, abs=0.01)
    assert tcz_out.streamline[0][1] == pytest.approx(-1.98, abs=0.01)
    assert tcz_out.streamline[0][2] == pytest.approx(22.95, abs=0.01)
    assert tcz_out.streamline[1][0] == pytest.approx(24.5, abs=0.01)
    assert tcz_out.streamline[1][1] == pytest.approx(48.90, abs=0.01)
    assert tcz_out.streamline[1][2] == pytest.approx(-1.98, abs=0.01)
    assert tcz_out.streamline[2][0] == pytest.approx(22.95, abs=0.01)
    assert tcz_out.streamline[2][1] == pytest.approx(24.5, abs=0.01)
    assert tcz_out.streamline[2][2] == pytest.approx(48.31, abs=0.01)
    assert tcz_out.streamline[3][0] == pytest.approx(-1.98, abs=0.01)
    assert tcz_out.streamline[3][1] == pytest.approx(22.95, abs=0.01)
    assert tcz_out.streamline[3][2] == pytest.approx(24.5, abs=0.01)
    assert tcz_out.streamline[4][0] == pytest.approx(-1.98, abs=0.01)
    assert tcz_out.streamline[4][1] == pytest.approx(45.90, abs=0.01)
    assert tcz_out.streamline[4][2] == pytest.approx(-1.98, abs=0.01)

    tcz_out.close(False)


def test_read_streamline_successfully():
    tcz = Tcz('tests/dicelib/mock/demo_fibers_read_streamline.tcz', 'r')
    assert tcz.read_streamline() == 4
    for x in range(4):
        for y in range(3):
            assert tcz.streamline[x][y] == 132.375


@pytest.mark.parametrize('input_number,expected_result', [
    (15.33334, 19371),
    (45.33334, 20907),
    (95.33334, 22005),
    (150.33334, 22707),
    (500.33334, 24529),
    (-15.33334, 52139),
    (-45.33334, 53675),
    (-95.33334, 54773),
    (-150.33334, 55475),
    (-500.33334, 57297),
])
def test_streamline_to_float16(input_number, expected_result):
    header_test = {
        'blur_core_extent': '1.1',
        'blur_gauss_extent': '2.2',
        'blur_spacing': '3.3',
        'epsilon': '0.4',
        'blur_gauss_min': '4.4',
        'representation': 'polyline',
        'datatype': 'Float32LE',
        'count': '999',
        'timestamp': '2040-01-01T00:00:00.000Z',
    }
    tcz = Tcz('tests/dicelib/mock/demo_fibers_write.tcz', 'w', header_test)
    tcz.n_pts = 4
    fake_streamline = np.full((4, 3), fill_value=input_number, dtype=np.float32)

    streamline_converted = tcz.compress_streamline(fake_streamline)
    for x in range(4):
        for y in range(3):
            assert streamline_converted[x][y] == expected_result


def test_create_with_invalid_format_will_throw_error():
    with pytest.raises(ValueError, match='Only ".tcz" files are supported for now.'):
        Tcz('demo01_fibers.xyz', 'r', )


def test_create_with_invalid_mode_will_throw_error():
    with pytest.raises(ValueError, match='"mode" must be either "r", "w" or "a"'):
        Tcz('tests/dicelib/mock/demo_fibers.tcz', 'fake-mode', )


def test_create_with_invalid_representation_param_will_throw_error():
    with pytest.raises(RuntimeError,
                       match='Problem parsing the header; field "representation" is not a valid value'):
        Tcz('tests/dicelib/mock/invalid/invalid_representation.tcz', 'r', )
