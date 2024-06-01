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
        'datatype': 'Float16',
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
