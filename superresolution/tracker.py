import ctypes
import sys
import os
import numpy
import numpy.ctypeslib


class Tracker(object):
    TRACKING_MOVING = 0
    TRACKING_STATIC = 1

    STRATEGY_BRUTE_FORCE = 0
    STRATEGY_KDTREE = 1

    track_input_type = {'dtype': [
        ('x', 'float64'),
        ('y', 'float64'),
        ('precision', 'float64'),
        ('frame', 'int64'),
        ('index', 'intp'),
        ('label', 'int64'),
        ('square_displacement', 'float64')
    ]}

    debug = True

    msd = None
    track = None

    def __init__(self):
        file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '_tracker.' + ('so' if sys.platform == 'linux' else 'dll'))

        old_cwd = os.getcwd()

        os.chdir(os.path.dirname(file))

        _track_so = ctypes.CDLL(file)

        os.chdir(old_cwd)

        _track_so.track.argtypes = (
            numpy.ctypeslib.ndpointer(**self.track_input_type),  # , flags='C_CONTIGUOUS'),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32
        )
        _track_so.track.restype = None

        _track_so.msd.argtypes = (
            numpy.ctypeslib.ndpointer(**self.track_input_type),  # , flags='C_CONTIGUOUS'),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float
        )
        _track_so.msd.restype = ctypes.c_float

        self._track_so = _track_so
        self._track = _track_so.track
        self._msd = _track_so.msd

    def track(self, transfer, maximum_displacement=1.0, memory=0, mode=None, strategy=None):
        if mode is None:
            mode = self.TRACKING_MOVING

        if strategy is None:
            strategy = self.STRATEGY_BRUTE_FORCE

        if len(transfer) == 0:
            raise RuntimeError('Empty data!')

        return self._track(transfer, len(transfer), maximum_displacement, memory, mode, strategy)

    def msd(self, transfer, micron_per_pixel=1.0, frames_per_second=1.0):
        if len(transfer) == 0:
            raise RuntimeError('Empty data!')

        return self._msd(transfer, len(transfer), micron_per_pixel, frames_per_second)

    def __del__(self):
        pass
        # if self.debug:
        #     return
        #     _handle = self._track_so._handle
        #     del self._track_so
        #     if sys.platform == 'linux':
        #         dl = ctypes.CDLL('libdl.so')
        #         dl.dlclose(_handle)

    @classmethod
    def empty_track_input_type(cls, count):
        return numpy.zeros(count, **cls.track_input_type)
