import matplotlib.pyplot as plt
import numpy as np
import pytest
from pylbo.visualisation.utils import (
    add_axis_label,
    add_textbox_to_axes,
    background_name_to_latex,
    ef_name_to_latex,
    ensure_attr_set,
)


def test_attr_set():
    class TestClass:
        def __init__(self):
            self.attr = None

    tc = TestClass()
    assert tc.attr is None
    with pytest.raises(AttributeError):
        ensure_attr_set(tc, "attr")


def test_ef_name_latex_cart():
    name = ef_name_to_latex("rho")
    assert name == r"$\rho$"
    name = ef_name_to_latex("v1")
    assert name == r"$v_x$"


def test_ef_name_latex_cart_real():
    name = ef_name_to_latex("rho", real_part=True)
    assert name == r"Re($\rho$)"
    name = ef_name_to_latex("v2", real_part=True)
    assert name == r"Re($v_y$)"


def test_ef_name_latex_cart_imag():
    name = ef_name_to_latex("T", real_part=False)
    assert name == r"Im($T$)"
    name = ef_name_to_latex("v3", real_part=False)
    assert name == r"Im($v_z$)"


def test_ef_name_latex_cyl():
    name = ef_name_to_latex("rho", geometry="cylindrical")
    assert name == r"$\rho$"
    name = ef_name_to_latex("v1", geometry="cylindrical")
    assert name == r"$v_r$"


def test_ef_name_latex_cyl_real():
    name = ef_name_to_latex("rho", geometry="cylindrical", real_part=True)
    assert name == r"Re($\rho$)"
    name = ef_name_to_latex("v2", geometry="cylindrical", real_part=True)
    assert name == r"Re($v_\theta$)"


def test_ef_name_latex_cyl_imag():
    name = ef_name_to_latex("rho", geometry="cylindrical", real_part=False)
    assert name == r"Im($\rho$)"
    name = ef_name_to_latex("v3", geometry="cylindrical", real_part=False)
    assert name == r"Im($v_z$)"


def test_bg_name_to_latex():
    assert background_name_to_latex("rho0") == r"$\rho_0$"
    assert background_name_to_latex("T0") == r"$T_0$"
    assert background_name_to_latex("lambdaT") == r"$\Lambda(T)$"
    assert background_name_to_latex("dB03") == r"$\partial B_{03}$"
    assert background_name_to_latex("v03") == r"$v_{03}$"


def test_add_txtbox_to_axes():
    _, ax = plt.subplots()
    bbox = add_textbox_to_axes(ax, "test", 0.5, 0.5)
    assert bbox is not None
    assert bbox in ax.get_children()


def test_add_axis_label_invalid():
    _, ax = plt.subplots()
    with pytest.raises(ValueError):
        add_axis_label(ax, "test", loc="upper middle")


def test_add_axis_label():
    _, ax = plt.subplots()
    label = add_axis_label(ax, "test", loc="top left")
    assert label is not None
    assert label in ax.get_children()


def test_add_axis_label_bold():
    _, ax = plt.subplots()
    label = add_axis_label(ax, "test", loc="top left", bold=True)
    assert label is not None
    assert label in ax.get_children()
    assert label.get_fontweight() == "bold"


def test_add_axis_label_coords_top_left():
    _, ax = plt.subplots()
    label = add_axis_label(ax, "test", loc="top left")
    assert label is not None
    assert label in ax.get_children()
    x, y = label.get_position()
    assert np.isclose(x, 0, atol=0.1)
    assert np.isclose(y, 1, atol=0.1)


def test_add_axis_label_coords_top_right():
    _, ax = plt.subplots()
    label = add_axis_label(ax, "test", loc="top right")
    assert label is not None
    assert label in ax.get_children()
    x, y = label.get_position()
    assert np.isclose(x, 1, atol=0.1)
    assert np.isclose(y, 1, atol=0.1)


def test_add_axis_label_coords_bottom_left():
    _, ax = plt.subplots()
    label = add_axis_label(ax, "test", loc="bottom left")
    assert label is not None
    assert label in ax.get_children()
    x, y = label.get_position()
    assert np.isclose(x, 0, atol=0.1)
    assert np.isclose(y, 0, atol=0.1)


def test_add_axis_label_coords_bottom_right():
    _, ax = plt.subplots()
    label = add_axis_label(ax, "test", loc="bottom right")
    assert label is not None
    assert label in ax.get_children()
    x, y = label.get_position()
    assert np.isclose(x, 1, atol=0.1)
    assert np.isclose(y, 0, atol=0.1)


def test_add_axis_label_coords_top_left_outside():
    _, ax = plt.subplots()
    label = add_axis_label(ax, "test", loc="top left", outside=True)
    assert label is not None
    assert label in ax.get_children()
    x, y = label.get_position()
    assert np.isclose(x, 0, atol=0.1)
    assert y > 1


def test_add_axis_label_coords_top_right_outside():
    _, ax = plt.subplots()
    label = add_axis_label(ax, "test", loc="top right", outside=True)
    assert label is not None
    assert label in ax.get_children()
    x, y = label.get_position()
    assert np.isclose(x, 1, atol=0.1)
    assert y > 1


def test_add_axis_label_coords_bottom_left_outside():
    _, ax = plt.subplots()
    label = add_axis_label(ax, "test", loc="bottom left", outside=True)
    assert label is not None
    assert label in ax.get_children()
    x, y = label.get_position()
    assert np.isclose(x, 0, atol=0.1)
    assert y < 0


def test_add_axis_label_coords_bottom_right_outside():
    _, ax = plt.subplots()
    label = add_axis_label(ax, "test", loc="bottom right", outside=True)
    assert label is not None
    assert label in ax.get_children()
    x, y = label.get_position()
    assert np.isclose(x, 1, atol=0.1)
    assert y < 0
