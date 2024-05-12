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
    assert tcz.header['streamline_representation'] == 'polyline'
    assert tcz.header['datatype'] == 'Float32LE'
    assert tcz.max_points == 1000
    assert len(tcz.streamline) == 1000
    assert type(tcz.streamline[0][0]) is float


def test_create_in_write_mode_successfully():
    header_test = {
        'blur_core_extent': '1.1',
        'blur_gauss_extent': '2.2',
        'blur_spacing': '3.3',
        'blur_gauss_min': '4.4',
        'streamline_representation': 'polyline',
        'datatype': 'Float32LE',
        'count': '999',
        'timestamp': '2040-01-01T00:00:00.000Z',
    }
    tcz = Tcz('tests/dicelib/mock/demo_fibers_write.tcz', 'w', header_test)
    assert tcz.streamline is None


def test_create_in_read_mode_no_streamline_representation_in_file_will_fall_back_to_polyline():
    tcz = Tcz('tests/dicelib/mock/demo_fibers_no_polyline.tcz', 'r', )
    assert tcz.header['streamline_representation'] == 'polyline'
    assert tcz.header['datatype'] == 'Float16'


def test_write_streamline_successfully():
    header_test = {
        'blur_core_extent': '1.1',
        'blur_gauss_extent': '2.2',
        'blur_spacing': '3.3',
        'blur_gauss_min': '4.4',
        'streamline_representation': 'polyline',
        'datatype': 'Float32LE',
        'count': '999',
        'timestamp': '2040-01-01T00:00:00.000Z',
    }
    tcz = Tcz('tests/dicelib/mock/demo_fibers_write_streamline.tcz', 'w', header_test)
    fake_streamline = np.full((4, 3), fill_value=132.364, dtype=np.float32)
    tcz.write_streamline(fake_streamline)


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
        'blur_gauss_min': '4.4',
        'streamline_representation': 'polyline',
        'datatype': 'Float32LE',
        'count': '999',
        'timestamp': '2040-01-01T00:00:00.000Z',
    }
    tcz = Tcz('tests/dicelib/mock/demo_fibers_write.tcz', 'w', header_test)

    fake_streamline = np.full((4, 3), fill_value=input_number, dtype=np.float32)

    streamline_converted = tcz.compress_streamline(fake_streamline)
    for x in range(4):
        for y in range(3):
            assert streamline_converted[x][y] == expected_result


@pytest.mark.parametrize('input_number,expected_result', [
    (19371, 15.3359375),
    (20907, 45.34375),
    (22005, 95.3125),
    (22707, 150.375),
    (24529, 500.25),
    (52139, -15.3359375),
    (53675, -45.34375),
    (54773, -95.3125),
    (55475, -150.375),
    (57297, -500.25),
])
def test_streamline_to_float32(input_number, expected_result):
    header_test = {
        'blur_core_extent': '1.1',
        'blur_gauss_extent': '2.2',
        'blur_spacing': '3.3',
        'blur_gauss_min': '4.4',
        'streamline_representation': 'polyline',
        'datatype': 'Float32LE',
        'count': '999',
        'timestamp': '2040-01-01T00:00:00.000Z',
    }
    tcz = Tcz('tests/dicelib/mock/demo_fibers_write.tcz', 'w', header_test)

    fake_streamline = np.full((4, 3), fill_value=input_number, dtype=np.uint16)

    streamline_converted = tcz.decompress_streamline(fake_streamline)
    for x in range(4):
        for y in range(3):
            assert streamline_converted[x][y] == expected_result


def test_create_with_invalid_format_will_throw_error():
    with pytest.raises(ValueError, match='Only ".tcz" files are supported for now.'):
        Tcz('demo01_fibers.xyz', 'r', )


def test_create_with_invalid_mode_will_throw_error():
    with pytest.raises(ValueError, match='"mode" must be either "r", "w" or "a"'):
        Tcz('tests/dicelib/mock/demo_fibers.tcz', 'fake-mode', )


@pytest.mark.parametrize('param', [
    'blur_core_extent',
    'blur_gauss_extent',
    'blur_spacing',
    'blur_gauss_min'
])
def test_create_with_missing_blur_params_will_throw_error(param):
    with pytest.raises(RuntimeError, match='Problem parsing the header; field "' + param + '" not found'):
        Tcz('tests/dicelib/mock/invalid/no_' + param + '.tcz', 'r', )


def test_create_with_invalid_streamline_representation_param_will_throw_error():
    with pytest.raises(RuntimeError,
                       match='Problem parsing the header; field "streamline_representation" is not a valid value'):
        Tcz('tests/dicelib/mock/invalid/invalid_streamline_representation.tcz', 'r', )


@pytest.mark.parametrize('param', [
    'blur_core_extent',
    'blur_gauss_extent',
    'blur_spacing',
    'blur_gauss_min'
])
def test_create_with_invalid_blur_params_will_throw_error(param):
    with pytest.raises(RuntimeError, match='"' + param + '" must be >= 0'):
        Tcz('tests/dicelib/mock/invalid/invalid_' + param + '.tcz', 'r', )
