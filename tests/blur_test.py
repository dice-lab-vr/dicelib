import pytest

from dicelib.blur import Blur


def test_create_blur_successfully():
    blur = Blur.from_header({
        'blur_core_extent': '1.1',
        'blur_gauss_extent': '2.2',
        'blur_spacing': '3.3',
        'blur_gauss_min': '4.4',
        'streamline_representation': 'polyline',
        'datatype': 'Float32LE',
        'count': '999',
        'timestamp': '2040-01-01T00:00:00.000Z',
    })

    assert blur.core_extent == 1.1
    assert blur.gauss_extent == 2.2
    assert blur.spacing == 3.3
    assert blur.gauss_min == 4.4


@pytest.mark.parametrize('header_with_negative_values,expected_error_message', [
    ({
         'blur_core_extent': '-1.1',
         'blur_gauss_extent': '2.2',
         'blur_spacing': '3.3',
         'blur_gauss_min': '4.4',
         'streamline_representation': 'polyline',
         'datatype': 'Float32LE',
         'count': '999',
         'timestamp': '2040-01-01T00:00:00.000Z',
     }, '"blur_core_extent" must be >= 0'),
    ({
         'blur_core_extent': '1.1',
         'blur_gauss_extent': '-2.2',
         'blur_spacing': '3.3',
         'blur_gauss_min': '4.4',
         'streamline_representation': 'polyline',
         'datatype': 'Float32LE',
         'count': '999',
         'timestamp': '2040-01-01T00:00:00.000Z',
     }, '"blur_gauss_extent" must be >= 0'),
    ({
         'blur_core_extent': '1.1',
         'blur_gauss_extent': '2.2',
         'blur_spacing': '-3.3',
         'blur_gauss_min': '4.4',
         'streamline_representation': 'polyline',
         'datatype': 'Float32LE',
         'count': '999',
         'timestamp': '2040-01-01T00:00:00.000Z',
     }, '"blur_spacing" must be >= 0'),
    ({
         'blur_core_extent': '1.1',
         'blur_gauss_extent': '2.2',
         'blur_spacing': '3.3',
         'blur_gauss_min': '-4.4',
         'streamline_representation': 'polyline',
         'datatype': 'Float32LE',
         'count': '999',
         'timestamp': '2040-01-01T00:00:00.000Z',
     }, '"blur_gauss_min" must be >= 0'),
])
def test_create_blur_with_negative_value_will_throw_error(header_with_negative_values, expected_error_message):
    with pytest.raises(RuntimeError, match=expected_error_message):
        Blur.from_header(header_with_negative_values)


@pytest.mark.parametrize('header_with_missing_values,expected_error_message', [
    ({
         'blur_gauss_extent': '2.2',
         'blur_spacing': '3.3',
         'blur_gauss_min': '4.4',
         'streamline_representation': 'polyline',
         'datatype': 'Float32LE',
         'count': '999',
         'timestamp': '2040-01-01T00:00:00.000Z',
     }, 'Problem parsing the header; field "blur_core_extent" not found'),
    ({
         'blur_core_extent': '1.1',
         'blur_spacing': '3.3',
         'blur_gauss_min': '4.4',
         'streamline_representation': 'polyline',
         'datatype': 'Float32LE',
         'count': '999',
         'timestamp': '2040-01-01T00:00:00.000Z',
     }, 'Problem parsing the header; field "blur_gauss_extent" not found'),
    ({
         'blur_core_extent': '1.1',
         'blur_gauss_extent': '2.2',
         'blur_gauss_min': '4.4',
         'streamline_representation': 'polyline',
         'datatype': 'Float32LE',
         'count': '999',
         'timestamp': '2040-01-01T00:00:00.000Z',
     }, 'Problem parsing the header; field "blur_spacing" not found'),
    ({
         'blur_core_extent': '1.1',
         'blur_gauss_extent': '2.2',
         'blur_spacing': '3.3',
         'streamline_representation': 'polyline',
         'datatype': 'Float32LE',
         'count': '999',
         'timestamp': '2040-01-01T00:00:00.000Z',
     }, 'Problem parsing the header; field "blur_gauss_min" not found'),
])
def test_create_blur_missing_core_extent_parameter_will_throw_error(header_with_missing_values, expected_error_message):
    with pytest.raises(RuntimeError, match=expected_error_message):
        Blur.from_header(header_with_missing_values)
