import pickle
from typing import List

import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseEvent, PickEvent
from pylbo.data_containers import LegolasDataSet


def pickle_dataseries_to_file(series, filepath):
    with open(filepath, "wb") as ostream:
        pickle.dump(series, ostream, pickle.HIGHEST_PROTOCOL)


def load_pickled_dataseries(filepath):
    with open(filepath, "rb") as istream:
        series = pickle.load(istream)
    return series


class FakeDataSet(LegolasDataSet):
    def __init__(self, datfile, seed=None):
        if seed is None:
            seed = 12345
        np.random.seed(seed)
        super().__init__(datfile)

        self._set_test_equilibria()
        self._set_test_eigenvalues()

    def _set_test_equilibria(self):
        for name in self.eq_names:
            x0 = np.random.randint(0, 10)
            x1 = np.random.randint(10, 25)
            self.equilibria[name] = np.linspace(x0, x1, self.gauss_gridpoints)
            if np.random.choice([True, False]):
                self.equilibria[name] = np.flip(self.equilibria[name])

    def _set_test_eigenvalues(self):
        self.eigenvalues = (
            np.random.randn(len(self.eigenvalues), 2).view(complex).flatten()
        )


class MockMouseEvent(MouseEvent):
    def __init__(self, button=1, canvas=None, axes=None, x=None, y=None):
        super().__init__(name="button_press_event", canvas=canvas, x=x, y=y)
        self.inaxes = axes
        self.button = button
        self.xdata = x
        self.ydata = y


class MockArtist:
    def __init__(self, ds, axes, figure):
        self.dataset = ds
        self.axes = axes
        self.figure = figure
        # xdata and ydata contain the (x, y) coordinates of all associated artists
        # that have a pick event (i.e. all eigenvalues)
        self._xdata = ds.eigenvalues.real
        self._ydata = ds.eigenvalues.imag

    def get_xdata(self):
        return np.atleast_1d(self._xdata)

    def get_ydata(self):
        return np.atleast_1d(self._ydata)


class MockPickEvent(PickEvent):
    def __init__(
        self,
        ds,
        mouse_x: float,
        mouse_y: float,
        button=1,
        axes=None,
        figure=None,
        ind: List = None,
    ):
        mouseevent = MockMouseEvent(
            button=button, canvas=figure.canvas, axes=axes, x=mouse_x, y=mouse_y
        )
        artist = MockArtist(ds, axes, figure)
        super().__init__(
            "pick_event", figure.canvas, mouseevent=mouseevent, artist=artist
        )
        # ind contains the indices of the selected eigenvalues
        idxs, _ = (
            ds.get_nearest_eigenvalues(mouse_x + mouse_y * 1j)
            if mouse_x is not None and mouse_y is not None
            else (None, None)
        )
        self.ind = ind if ind is not None else idxs


class MockKeyEvent(KeyEvent):
    def __init__(self, key, figure=None):
        super().__init__(name="key_press_event", key=key, canvas=figure.canvas)
