from typing import TypeVar

T = TypeVar("T")

class SizeLimitedDict(dict):
    def __init__(self, max_size: int, *args, **kwargs):
        self._max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self._max_size:
            oldest = next(iter(self))
            del self[oldest]
            