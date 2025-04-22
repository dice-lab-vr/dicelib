class Blur:

    def __init__(self, core_extent, gauss_extent, gauss_min, spacing):
        self._core_extent = core_extent
        self._gauss_extent = gauss_extent
        self._gauss_min = gauss_min
        self._spacing = spacing

    @classmethod
    def from_header(cls, data):

        if 'blur_core_extent' not in data:
            raise RuntimeError('Problem parsing the header; field "blur_core_extent" not found')
        if 'blur_gauss_extent' not in data:
            raise RuntimeError('Problem parsing the header; field "blur_gauss_extent" not found')
        if 'blur_spacing' not in data:
            raise RuntimeError('Problem parsing the header; field "blur_spacing" not found')
        if 'blur_gauss_min' not in data:
            raise RuntimeError('Problem parsing the header; field "blur_gauss_min" not found')

        if float(data['blur_core_extent']) < 0:
            raise RuntimeError('"blur_core_extent" must be >= 0')
        if float(data['blur_gauss_extent']) < 0:
            raise RuntimeError('"blur_gauss_extent" must be >= 0')
        if float(data['blur_gauss_min']) < 0:
            raise RuntimeError('"blur_gauss_min" must be >= 0')
        if float(data['blur_spacing']) < 0:
            raise RuntimeError('"blur_spacing" must be >= 0')

        return cls(
            float(data['blur_core_extent']),
            float(data['blur_gauss_extent']),
            float(data['blur_gauss_min']),
            float(data['blur_spacing'])
        )

    @property
    def core_extent(self):
        return self._core_extent

    @property
    def gauss_extent(self):
        return self._gauss_extent

    @property
    def gauss_min(self):
        return self._gauss_min

    @property
    def spacing(self):
        return self._spacing
