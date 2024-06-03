import pytest

from dicelib.header import Header


def test_create_header_successfully():
    data = {
        'blur_core_extent': '1.1',
        'blur_gauss_extent': '2.2',
        'blur_spacing': '3.3',
        'blur_gauss_min': '4.4',
        'segment_len': 5.5,
        'epsilon': 0.3,
        'representation': 'spline',
        'timestamp': '2040-01-01T00:00:00.000Z',
        'count': 1,
        'file': 999,
    }

    header = Header.for_tcz(data)

    assert header.as_dict() == {
        'blur_core_extent': 1.1,
        'blur_gauss_extent': 2.2,
        'blur_gauss_min': 4.4,
        'blur_spacing': 3.3,
        'count': 1,
        'file': 999,
        'timestamp': '2040-01-01T00:00:00.000Z',
        'datatype': 'Float16',
        'epsilon': 0.3,
        'representation': 'spline',
        'segment_len': 5.5
    }


def test_create_with_no_segment_len_will_automatically_it():
    data = {
        'blur_core_extent': '1.1',
        'blur_gauss_extent': '2.2',
        'blur_spacing': '3.3',
        'blur_gauss_min': '4.4',
        'epsilon': 0.3,
        'representation': 'spline',
        'datatype': 'Float16',
        'timestamp': '2040-01-01T00:00:00.000Z',
        'count': 1,
        'file': 999,
    }

    header = Header.for_tcz(data)

    assert header.segment_len == 0.5


def test_create_with_no_epsilon_will_automatically_it():
    data = {
        'blur_core_extent': '1.1',
        'blur_gauss_extent': '2.2',
        'blur_spacing': '3.3',
        'blur_gauss_min': '4.4',
        'segment_len': 1.2,
        'representation': 'spline',
        'timestamp': '2040-01-01T00:00:00.000Z',
        'count': 1,
        'file': 999,
    }

    header = Header.for_tcz(data)

    assert header.epsilon == 0.3


@pytest.mark.parametrize('representation', [
    'rdp',
    'spline',
    'polyline',
])
def test_create_with_allowed_representations_successfully(representation):
    data = {
        'blur_core_extent': '1.1',
        'blur_gauss_extent': '2.2',
        'blur_spacing': '3.3',
        'blur_gauss_min': '4.4',
        'representation': representation,
        'segment_len': 5.5,
        'epsilon': 0.3,
        'timestamp': '2040-01-01T00:00:00.000Z',
        'count': 1,
        'file': 999,
    }

    header = Header.for_tcz(data)
    assert header.representation == representation


def test_create_header_with_no_representation_will_fallback_to_polyline():
    data = {
        'blur_core_extent': '1.1',
        'blur_gauss_extent': '2.2',
        'blur_spacing': '3.3',
        'blur_gauss_min': '4.4',
        'segment_len': 5.5,
        'epsilon': 0.3,
        'timestamp': '2040-01-01T00:00:00.000Z',
        'count': 1,
        'file': 999,
    }

    header = Header.for_tcz(data)
    assert header.representation == 'polyline'


def test_create_with_no_count_will_throw_exception():
    with pytest.raises(RuntimeError, match='Problem parsing the header; field "count" not found'):
        Header.for_tcz({
            'blur_core_extent': '1.1',
            'blur_gauss_extent': '2.2',
            'blur_spacing': '3.3',
            'blur_gauss_min': '4.4',
            'segment_len': 5.5,
            'epsilon': 0.3,
            'timestamp': '2040-01-01T00:00:00.000Z',
            'file': 999,
        })


def test_create_with_no_file_will_throw_exception():
    with pytest.raises(RuntimeError, match='Problem parsing the header; field "file" not found'):
        Header.for_tcz({
            'blur_core_extent': '1.1',
            'blur_gauss_extent': '2.2',
            'blur_spacing': '3.3',
            'blur_gauss_min': '4.4',
            'segment_len': 5.5,
            'epsilon': 0.3,
            'count': 1,
            'timestamp': '2040-01-01T00:00:00.000Z',
        })


def test_multiple_count_value_will_throw_exception():
    with pytest.raises(RuntimeError, match='Problem parsing the header; field "count" has multiple values'):
        Header.for_tcz({
            'blur_core_extent': '1.1',
            'blur_gauss_extent': '2.2',
            'blur_spacing': '3.3',
            'blur_gauss_min': '4.4',
            'segment_len': 5.5,
            'epsilon': 0.3,
            'count': [1, 2],
            'timestamp': '2040-01-01T00:00:00.000Z',
        })


def test_multiple_file_value_will_throw_exception():
    with pytest.raises(RuntimeError, match='Problem parsing the header; field "file" has multiple values'):
        Header.for_tcz({
            'blur_core_extent': '1.1',
            'blur_gauss_extent': '2.2',
            'blur_spacing': '3.3',
            'blur_gauss_min': '4.4',
            'segment_len': 5.5,
            'epsilon': 0.3,
            'count': 1,
            'file': [1, 2],
            'timestamp': '2040-01-01T00:00:00.000Z',
        })
