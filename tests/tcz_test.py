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

def test_create_successfully_in_read_mode():
    tcz = Tcz('tests/dicelib/mock/demo_fibers.tcz', 'r', )
    assert isinstance(tcz, Tcz)
    print(tcz.header)
    assert tcz.header['blur_core_extent'] == 12.3
    assert tcz.header['blur_gauss_extent'] == 34.4
    assert tcz.header['blur_spacing'] == 23.1
    assert tcz.header['blur_gauss_min'] == 34.0
    assert tcz.header['streamline_representation'] == 'polyline'
    assert tcz.header['datatype'] == 'Float32LE'


def test_create_in_read_mode_no_streamline_representation_in_file_will_fall_back_to_polyline():
    tcz = Tcz('tests/dicelib/mock/demo_fibers_no_polyline.tcz', 'r', )
    assert tcz.header['streamline_representation'] == 'polyline'
    assert tcz.header['datatype'] == 'Float16'


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
#
#
# @pytest.mark.parametrize('param', [
#     'blur_core_extent',
#     'blur_gauss_extent',
#     'blur_spacing',
#     'blur_gauss_min'
# ])
# def test_create_with_invalid_blur_params_will_throw_error(param):
#     with pytest.raises(RuntimeError, match='"' + param + '" must be >= 0'):
#         Tcz('tests/dicelib/mock/invalid/invalid_' + param + '.tcz', 'r', )
