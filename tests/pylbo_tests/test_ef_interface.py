import logging
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pylbo
from pylbo.testing import MockKeyEvent, MockPickEvent
from pylbo.utilities.logger import pylboLogger
from pylbo.visualisation.legend_handler import _get_legend_handles

MODE1 = 0.1 + 0.62j
MODE2 = 0.1 + 0.58j
MODE3 = 0.1 + 0.54j
MODES = [MODE1, MODE2, MODE3]

EV1 = -0.00191999 + 0.62143192j
EV2 = -0.00187198 + 0.58010620j
EV3 = -0.00201184 + 0.54839281j
ACTUAL_EVS = [EV1, EV2, EV3]


def _create_ef_spectrum_plot(ds):
    p = pylbo.plot_spectrum(ds)
    p.add_eigenfunctions()
    return p


def _get_efs(ds):
    (efs,) = ds.get_eigenfunctions(MODE1)
    return efs


def _create_mockpickevent(p, x, y, button, ind=None):
    return MockPickEvent(
        mouse_x=x,
        mouse_y=y,
        button=button,
        ds=p.dataset,
        axes=p.ax,
        figure=p.fig,
        ind=ind,
    )


def _create_mockkeyevent(p, key):
    return MockKeyEvent(key=key, figure=p.fig)


def _create_single_pickevent(p, x, y, button, ind=None):
    event = _create_mockpickevent(p, x, y, button, ind)
    p.ef_handler.on_point_pick(event)


def _create_single_keyevent(p, key):
    event = _create_mockkeyevent(p, key)
    p.ef_handler.on_key_press(event)


def _get_ef_ax_title(p):
    return p.ef_handler.axis.get_title()


def _get_xydata_from_ef_plot(p):
    axis = p.ef_handler.axis
    line = [
        child
        for child in axis.get_children()
        if isinstance(child.get_label(), str) and "omega" in child.get_label()
    ][0]
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    return xdata, ydata


def test_ef_regular_plot(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    assert p.ef_handler is not None
    assert isinstance(p.ef_ax, plt.Axes)
    assert p.ef_handler.get_selected_idxs() == {}


def test_ef_left_click_single(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    _create_single_pickevent(p, x=0.1, y=0.62, button=1)
    selected_dict = p.ef_handler.get_selected_idxs()
    assert len(selected_dict) == 1
    artist_dict = selected_dict.get(ds_v200_mri_efs, None)
    assert artist_dict is not None
    for value in artist_dict.values():
        assert isinstance(value, plt.Line2D)


def test_ef_left_click_multiple(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    for x, y in zip([0.1, 0.1, 0.1], [0.62, 0.58, 0.54]):
        _create_single_pickevent(p, x=x, y=y, button=1)
    artist_dict = p.ef_handler.get_selected_idxs().get(ds_v200_mri_efs, None)
    assert artist_dict is not None
    assert len(artist_dict) == 3
    for value in artist_dict.values():
        assert isinstance(value, plt.Line2D)


def test_ef_left_click_already_selected(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    _create_single_pickevent(p, x=0.1, y=0.62, button=1)
    artist_dict = p.ef_handler.get_selected_idxs().get(ds_v200_mri_efs, None)
    assert artist_dict is not None
    assert len(artist_dict) == 1
    _create_single_pickevent(p, x=0.1, y=0.62, button=1)
    artist_dict = p.ef_handler.get_selected_idxs().get(ds_v200_mri_efs, None)
    assert artist_dict is not None
    assert len(artist_dict) == 1


def test_ef_left_click_not_in_subset(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    _create_single_pickevent(p, x=0.1, y=0.62, button=1)
    artist_dict = p.ef_handler.get_selected_idxs().get(ds_v200_mri_efs, None)
    assert artist_dict is not None
    assert len(artist_dict) == 1
    _create_single_pickevent(p, x=0.1, y=0.16, button=1)
    artist_dict = p.ef_handler.get_selected_idxs().get(ds_v200_mri_efs, None)
    assert artist_dict is not None
    assert len(artist_dict) == 1


def test_ef_left_click_overlap(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    idxs, _ = ds_v200_mri_efs.get_nearest_eigenvalues([MODE1, MODE2])
    _create_single_pickevent(p, x=0.1, y=0.62, button=1, ind=idxs)
    artist_dict = p.ef_handler.get_selected_idxs().get(ds_v200_mri_efs, None)
    assert artist_dict is not None
    for value in artist_dict.values():
        assert isinstance(value, plt.Line2D)
    assert idxs[0] == list(artist_dict.keys())


def test_ef_right_click(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    _create_single_pickevent(p, x=0.1, y=0.62, button=1)
    artist_dict = p.ef_handler.get_selected_idxs().get(ds_v200_mri_efs, None)
    assert artist_dict is not None

    _create_single_pickevent(p, x=0.1, y=0.62, button=3)
    assert p.ef_handler.get_selected_idxs() == {}
    artist_dict = p.ef_handler.get_selected_idxs().get(ds_v200_mri_efs, None)
    assert artist_dict is None


def test_ef_left_click_draws(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    # check that no eigenfunctions are drawn yet
    for child in p.ef_handler.axis.get_children():
        assert not isinstance(child, plt.Line2D)
        assert not isinstance(child, mpl.legend.Legend)

    _create_single_pickevent(p, x=0.1, y=0.62, button=1)
    children = set(p.ef_handler.axis.get_children())
    # check that eigenfunctions are drawn, including the x=0 and y=0 lines
    assert Counter(isinstance(child, plt.Line2D) for child in children)[True] == 3
    # check we have a legend
    assert any(isinstance(child, mpl.legend.Legend) for child in children)
    assert p.ef_handler.get_name_of_drawn_eigenfunction() == "rho"
    title = _get_ef_ax_title(p)
    assert "Re" in title or "Im" in title


def test_ef_click_event_not_in_axes(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    event = _create_mockpickevent(p, x=0.1, y=0.62, button=1)
    event.mouseevent.inaxes = p.ef_handler.axis
    p.ef_handler.on_point_pick(event)
    assert p.ef_handler.get_selected_idxs() == {}
    assert p.ef_handler.get_selected_idxs().get(ds_v200_mri_efs, None) is None


def test_ef_click_on_legend_item(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    p.add_continua()
    event = _create_mockpickevent(p, x=0.1, y=0.62, button=1)
    # set artist equal to clickable first legend item instead of datapoints
    event.artist = _get_legend_handles(p.ef_handler.spec_axis.get_legend())[0]
    p.ef_handler.on_point_pick(event)
    # should do nothing
    assert p.ef_handler.get_selected_idxs() == {}
    assert p.ef_handler.get_selected_idxs().get(ds_v200_mri_efs, None) is None


def test_ef_click_on_random_artist(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    p.add_continua()
    event = _create_mockpickevent(p, x=0.1, y=0.62, button=1)
    # clicking on the legend itself should do nothing
    event.artist = p.ef_handler.spec_axis.get_legend()
    p.ef_handler.on_point_pick(event)
    assert p.ef_handler.get_selected_idxs() == {}
    assert p.ef_handler.get_selected_idxs().get(ds_v200_mri_efs, None) is None


def test_ef_get_artist_data(ds_v200_mri_efs):
    from pylbo.visualisation.eigenfunctions.eigfunc_interface import get_artist_data

    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    p.add_continua()
    artist = _get_legend_handles(p.ef_handler.spec_axis.get_legend())[0]
    xdata, ydata = get_artist_data(artist)
    assert isinstance(xdata, np.ndarray)
    assert xdata.ndim == 1
    assert isinstance(ydata, np.ndarray)
    assert ydata.ndim == 1


def test_ef_keypress_clear_selection(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    _create_single_pickevent(p, x=0.1, y=0.62, button=1)
    assert p.ef_handler.get_selected_idxs().get(ds_v200_mri_efs, None) is not None
    _create_single_keyevent(p, key="d")
    assert p.ef_handler.get_selected_idxs() == {}
    assert p.ef_handler.get_selected_idxs().get(ds_v200_mri_efs, None) is None


def test_ef_keypress_switch_re_im(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    efs = _get_efs(ds_v200_mri_efs)
    _create_single_pickevent(p, x=0.1, y=0.62, button=1)
    assert "Re" in _get_ef_ax_title(p)
    _, ydata = _get_xydata_from_ef_plot(p)
    assert np.all(ydata == efs.get("rho").real)

    _create_single_keyevent(p, key="i")
    assert "Im" in _get_ef_ax_title(p)
    _, ydata = _get_xydata_from_ef_plot(p)
    assert np.all(ydata == efs.get("rho").imag)


def test_ef_keypress_select_next(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    _create_single_pickevent(p, x=0.1, y=0.62, button=1)
    assert "rho" in _get_ef_ax_title(p)
    _create_single_keyevent(p, key="up")
    assert "v_r" in _get_ef_ax_title(p)

    efs = _get_efs(ds_v200_mri_efs)
    assert np.all(_get_xydata_from_ef_plot(p)[1] == efs.get("v1").real)


def test_ef_keypress_select_next_end(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    (efs,) = ds_v200_mri_efs.get_eigenfunctions(MODE1)
    fnames = p.ef_handler._function_names
    last_key = fnames[-1]
    p.ef_handler._selected_name_idx = len(fnames) - 1
    _create_single_pickevent(p, x=0.1, y=0.62, button=1)
    assert "perp" in _get_ef_ax_title(p) if "perp" in last_key else True
    assert np.all(_get_xydata_from_ef_plot(p)[1] == efs.get(last_key).real)
    _create_single_keyevent(p, key="up")
    assert fnames[0] in _get_ef_ax_title(p)
    assert np.all(_get_xydata_from_ef_plot(p)[1] == efs.get(fnames[0]).real)


def test_ef_keypress_select_prev(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    _create_single_pickevent(p, x=0.1, y=0.62, button=1)
    assert "rho" in _get_ef_ax_title(p)
    [_create_single_keyevent(p, key="up") for _ in range(4)]
    assert "T" in _get_ef_ax_title(p)
    _create_single_keyevent(p, key="down")
    assert "v_z" in _get_ef_ax_title(p)
    efs = _get_efs(ds_v200_mri_efs)
    assert np.all(_get_xydata_from_ef_plot(p)[1] == efs.get("v3").real)


def test_ef_keypress_select_prev_first(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    (efs,) = ds_v200_mri_efs.get_eigenfunctions(MODE1)
    fnames = p.ef_handler._function_names
    last_key = fnames[-1]
    _create_single_pickevent(p, x=0.1, y=0.62, button=1)
    assert fnames[0] in _get_ef_ax_title(p)
    _create_single_keyevent(p, key="down")
    assert "perp" in _get_ef_ax_title(p) if "perp" in last_key else True
    assert np.all(_get_xydata_from_ef_plot(p)[1] == efs.get(last_key).real)


def test_ef_keypress_retransform(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    efs = _get_efs(ds_v200_mri_efs)
    _create_single_pickevent(p, x=0.1, y=0.62, button=1)
    assert "rho" in _get_ef_ax_title(p)
    _create_single_keyevent(p, key="up")
    assert "$v_r$" in _get_ef_ax_title(p)
    assert np.all(_get_xydata_from_ef_plot(p)[1] == efs.get("v1").real)
    _create_single_keyevent(p, key="t")
    assert "$rv_r$" in _get_ef_ax_title(p)
    efgrid = ds_v200_mri_efs.ef_grid
    assert np.all(_get_xydata_from_ef_plot(p)[1] == efgrid * efs.get("v1").real)


def test_ef_keypress_save_ev_selection(ds_v200_mri_efs, tmpdir):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    for mode in MODES:
        _create_single_pickevent(p, x=mode.real, y=mode.imag, button=1)
    p.ef_handler.savedir = tmpdir
    _create_single_keyevent(p, key="a")
    expected_file = tmpdir / f"{ds_v200_mri_efs.datfile.name}"
    assert expected_file.with_suffix(".npy").exists()
    expected_file.with_suffix(".npy").unlink()


def test_ef_keypress_save_ev_selection_none(ds_v200_mri_efs, tmpdir):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    p.ef_handler.savedir = tmpdir
    _create_single_keyevent(p, key="a")
    expected_file = tmpdir / f"{ds_v200_mri_efs.datfile.name}"
    assert not expected_file.with_suffix(".npy").exists()


def test_ef_keypress_save_ev_idx_selection(ds_v200_mri_efs, tmpdir):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    for mode in MODES:
        _create_single_pickevent(p, x=mode.real, y=mode.imag, button=1)
    p.ef_handler.savedir = tmpdir
    _create_single_keyevent(p, key="j")
    expected_file = tmpdir / f"{ds_v200_mri_efs.datfile.name}"
    assert expected_file.with_suffix(".npy").exists()
    expected_file.with_suffix(".npy").unlink()


def test_ef_keypress_save_ev_idx_selection_none(ds_v200_mri_efs, tmpdir):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    p.ef_handler.savedir = tmpdir
    _create_single_keyevent(p, key="j")
    expected_file = tmpdir / f"{ds_v200_mri_efs.datfile.name}"
    assert not expected_file.with_suffix(".npy").exists()


def test_ef_tooltip_is_displayed(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    assert any(
        isinstance(child, mpl.table.Table) for child in p.ef_handler.axis.get_children()
    )
    _create_single_pickevent(p, x=0.1, y=0.62, button=1)
    assert not any(
        isinstance(child, mpl.table.Table) for child in p.ef_handler.axis.get_children()
    )
    _create_single_keyevent(p, key="d")
    assert any(
        isinstance(child, mpl.table.Table) for child in p.ef_handler.axis.get_children()
    )


def test_ef_print_selected_eigenvalues(ds_v200_mri_efs, caplog):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    for mode in MODES:
        _create_single_pickevent(p, x=mode.real, y=mode.imag, button=1)
    with caplog.at_level(logging.INFO, logger=pylboLogger.name):
        _create_single_keyevent(p, key="w")
    text = caplog.text
    text = text[text.find("[") + 1 : text.find("]")].replace("  ", " ").split(" ")
    for ev, txt in zip(ACTUAL_EVS, text):
        txt_val = complex(txt.replace(" ", ""))
        assert np.isclose(ev, txt_val)


def test_ef_print_evs_nothing_selected(ds_v200_mri_efs, caplog):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    with caplog.at_level(logging.INFO, logger=pylboLogger.name):
        _create_single_keyevent(p, key="w")
    assert caplog.text == ""


def test_ef_print_nonzeroes(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    _create_single_pickevent(p, x=0.1, y=0.62, button=1)
    _create_single_keyevent(p, key="n")


def test_ef_print_nonzeroes_nothing_selected(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    _create_single_keyevent(p, key="n")


def test_toggle_ef_subset(ds_v200_mri_efs):
    p = _create_ef_spectrum_plot(ds_v200_mri_efs)
    _create_single_keyevent(p, key="e")
    children = p.ax.get_children()
    circle = [child for child in children if isinstance(child, mpl.patches.Circle)]
    circle = circle[0] if circle else None
    assert circle is not None
    assert np.isclose(circle.radius, ds_v200_mri_efs.header["ef_subset_radius"])
    assert np.isclose(circle.center[0], ds_v200_mri_efs.header["ef_subset_center"].real)
    assert np.isclose(circle.center[1], ds_v200_mri_efs.header["ef_subset_center"].imag)
