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
    assert tcz.segment_len == pytest.approx(0.7, abs=0.01)
    assert tcz.epsilon == pytest.approx(0.4, abs=0.01)
    assert tcz.max_points == 1000
    assert len(tcz.streamline) == 1000
    assert type(tcz.streamline[0][0]) is float


def test_create_in_read_mode_no_segment_len_will_setup_segment_len_automatically():
    tcz = Tcz('tests/dicelib/mock/demo_fibers_no_segment_len.tcz', 'r', None, 1000)
    assert tcz.segment_len == 0.5


def test_create_in_read_mode_streamline_spline_successfully():
    tcz = Tcz('tests/dicelib/mock/demo_fibers_streamline_spline.tcz', 'r', None, 1000)
    assert tcz.header['representation'] == 'spline'


def test_create_in_read_mode_streamline_rdp_successfully():
    tcz = Tcz('tests/dicelib/mock/demo_fibers_streamline_rdp.tcz', 'r', None, 1000)
    assert tcz.header['representation'] == 'rdp'


def test_create_in_write_mode_successfully():
    header_test = {
        'blur_core_extent': '1.1',
        'blur_gauss_extent': '2.2',
        'blur_spacing': '3.3',
        'blur_gauss_min': '4.4',
        'epsilon': '0.005',
        'segment_len': '0.8',
        'representation': 'spline',
        'datatype': 'Float32LE',
        'count': '999',
        'timestamp': '2040-01-01T00:00:00.000Z',
    }
    tcz = Tcz('tests/dicelib/mock/demo_fibers_write.tcz', 'w', header_test)
    assert tcz.segment_len == pytest.approx(0.8, abs=0.01)
    assert tcz.epsilon == pytest.approx(0.005, abs=0.001)
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
    n_points = 4
    fake_streamline = np.full((n_points, 3), fill_value=132.364, dtype=np.float32)
    tcz.write_streamline(fake_streamline, n_points)


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
    n_points = 6
    fake_streamline = np.array([
        [2.1, 3.2, 1.3],
        [1.1, 2.2, 3.3],
        [1.0, 0.5, 5.4],
        [1.9, 8.1, 1.1],
        [0.8, 0.2, 0.1],
        [6.8, 4.2, 5.1],
    ], dtype=np.float32)
    tcz_out.write_streamline(fake_streamline, n_points)
    tcz_out.close()


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
        'epsilon': '0.005',
        'blur_gauss_min': '4.4',
        'representation': 'polyline',
        'datatype': 'Float32LE',
        'count': '999',
        'timestamp': '2040-01-01T00:00:00.000Z',
    }
    tcz = Tcz('tests/dicelib/mock/demo_fibers_write.tcz', 'w', header_test)
    fake_streamline = np.full((4, 3), fill_value=input_number, dtype=np.float32)
    streamline_converted = tcz.compress_streamline(fake_streamline, 4)
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
