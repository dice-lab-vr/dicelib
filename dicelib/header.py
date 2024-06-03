from dicelib.blur import Blur


class Header:

    def __init__(self, blur, segment_len, epsilon, representation, datatype, timestamp, count, file):
        self.blur = blur
        self.segment_len = segment_len
        self.epsilon = epsilon
        self.representation = representation
        self.datatype = datatype
        self.timestamp = timestamp
        self.count = count
        self.file = file

    @classmethod
    def for_tcz(cls, data):
        blur = Blur.from_header(data)

        if 'epsilon' not in data:
            epsilon = 0.3
        else:
            epsilon = float(data['epsilon'])

        if 'segment_len' not in data:
            segment_len = 0.5
        else:
            segment_len = float(data['segment_len'])

        if 'representation' not in data:
            representation = 'polyline'
        else:
            representation = data['representation']

        if representation not in ['polyline', 'spline', 'rdp']:
            raise RuntimeError('Problem parsing the header; field "representation" is not a valid value')

        if 'count' not in data:
            raise RuntimeError('Problem parsing the header; field "count" not found')

        if type(data['count']) is list:
            raise RuntimeError('Problem parsing the header; field "count" has multiple values')

        if 'file' not in data:
            raise RuntimeError('Problem parsing the header; field "file" not found')
        if type(data['file']) is list:
            raise RuntimeError('Problem parsing the header; field "file" has multiple values')

        return cls(
            blur,
            segment_len,
            epsilon,
            representation,
            'Float16',
            data['timestamp'],
            data['count'],
            data['file']
        )

    def as_dict(self):
        return {
            'blur_core_extent': self.blur.core_extent,
            'blur_gauss_extent': self.blur.gauss_extent,
            'blur_spacing': self.blur.spacing,
            'blur_gauss_min': self.blur.gauss_min,
            'epsilon': self.epsilon,
            'representation': self.representation,
            'segment_len': self.segment_len,
            'datatype': self.datatype,
            'timestamp': self.timestamp,
            'count': self.count,
            'file': self.file
        }
