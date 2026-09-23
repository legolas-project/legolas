import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pylbo
import pytest
from pylbo.exceptions import BackgroundNotPresent, EigenfunctionsNotPresent
from pylbo.utilities.toolbox import get_axis_geometry
from pylbo.visualisation.spectra.spectrum_single import SingleSpectrumPlot


def test_spectrum_plot_invalid_data(series_v112):
    with pytest.raises(TypeError):
        pylbo.plot_spectrum(series_v112)


def test_spectrum_plot(ds_v112):
    pylbo.plot_spectrum(ds_v112)


def test_spectum_plot_continua(ds_v112):
    p = pylbo.plot_spectrum(ds_v112)
    p.add_continua()
    labels = p.ax.get_legend_handles_labels()[1]
    assert len(labels) > 2
    assert getattr(p, "c_handler", None) is not None


def test_spectrum_plot_eigenfunctions(ds_v112):
    p = pylbo.plot_spectrum(ds_v112)
    p.add_eigenfunctions()
    assert p.ax.get_subplotspec().get_geometry()[0:2] == (1, 2)
    assert getattr(p, "ef_handler", None) is not None
    assert getattr(p, "ef_ax", None) is not None


def test_spectrum_plot_continua_and_eigenfunctions(ds_v112):
    p = pylbo.plot_spectrum(ds_v112)
    p.add_continua()
    p.add_eigenfunctions()
    labels = p.ax.get_legend_handles_labels()[1]
    assert len(labels) > 2
    for attr in ("c_handler", "ef_handler", "ef_ax"):
        assert getattr(p, attr, None) is not None


def test_spectrum_plot_eigenfunctions_notpresent(ds_v100):
    p = pylbo.plot_spectrum(ds_v100)
    with pytest.raises(EigenfunctionsNotPresent):
        p.add_eigenfunctions()


def test_spectrum_plot_set_x_scaling(ds_v112):
    p = pylbo.plot_spectrum(ds_v112)
    p.draw()
    scale = 2.2
    xlims = np.array(p.ax.get_xlim(), dtype=float)
    p.set_x_scaling(scale)
    assert np.allclose(scale * xlims, p.ax.get_xlim())


def test_spectrum_plot_set_y_scaling(ds_v112):
    p = pylbo.plot_spectrum(ds_v112)
    p.draw()
    scale = 5.8
    ylims = np.array(p.ax.get_ylim(), dtype=float)
    p.set_y_scaling(scale)
    assert np.allclose(scale * ylims, p.ax.get_ylim())


def test_spectrum_plot_set_scaling(ds_v112):
    p = pylbo.plot_spectrum(ds_v112)
    p.draw()
    xscale = 3.5
    yscale = -2.1
    xlims = np.array(p.ax.get_xlim(), dtype=float)
    ylims = np.flip(np.array(p.ax.get_ylim(), dtype=float))  # flip: negative yscale
    p.set_x_scaling(xscale)
    p.set_y_scaling(yscale)
    assert np.allclose(xscale * xlims, p.ax.get_xlim())
    assert np.allclose(yscale * ylims, p.ax.get_ylim())


def test_spectrum_plot_set_scaling_efs_conts(ds_v112):
    p = pylbo.plot_spectrum(ds_v112)
    p.draw()
    p.add_continua()
    p.add_eigenfunctions()
    xscale = -12.4
    xlims = np.flip(np.array(p.ax.get_xlim(), dtype=float))
    p.set_x_scaling(xscale)
    assert np.allclose(xscale * xlims, p.ax.get_xlim())


def test_spectrum_plot_save(tmpdir, ds_v112):
    p = pylbo.plot_spectrum(ds_v112)
    filepath = tmpdir / "testfig.png"
    p.save(filepath)
    assert filepath.is_file()


def test_spectrum_plot_custom_figure_efs(ds_v112):
    fig, axes = plt.subplots(1, 2)
    p = pylbo.plot_spectrum(ds_v112, custom_figure=(fig, axes[0]))
    p.add_eigenfunctions()
    assert len(fig.get_axes()) == 3


def test_spectrum_plot_custom_figure(ds_v112):
    fig, ax = plt.subplots(1)
    p = pylbo.plot_spectrum(ds_v112, custom_figure=(fig, ax))
    assert p.fig == fig
    assert p.ax == ax


def test_spectrum_plot_figsize(ds_v112):
    p = pylbo.plot_spectrum(ds_v112, figsize=(8, 8))
    assert np.all(p.fig.get_size_inches() == (8, 8))
    assert np.all(p.figsize == (8, 8))


def test_spectrum_plot_figlabel(ds_v112):
    fig = plt.figure("testfig")
    ax = fig.add_subplot(111)
    p = pylbo.plot_spectrum(ds_v112, custom_figure=(fig, ax))
    assert p.figure_id == "testfig"


def test_spectrum_plot_nobg_continua(ds_v200_tear_nobg):
    p = pylbo.plot_spectrum(ds_v200_tear_nobg)
    p.add_continua()
    assert getattr(p, "c_handler") is None


def test_multispectrum_plot_invalid_data(ds_v112):
    with pytest.raises(TypeError):
        pylbo.plot_spectrum_multi(ds_v112, xdata="k2")


def test_multispectrum_plot_xdata_invalid_size(series_v112):
    with pytest.raises(ValueError):
        pylbo.plot_spectrum_multi(series_v112, xdata=[5, 5])


def test_multispectrum_plot_xdata_invalid_type(series_v112):
    with pytest.raises(TypeError):
        pylbo.plot_spectrum_multi(series_v112, xdata=3.0 + 1j)


def test_multispectrum_plot_xdata_invalid_key(series_v112):
    with pytest.raises(ValueError):
        pylbo.plot_spectrum_multi(series_v112, xdata="k4")


def test_multispectrum_plot_xdata_str(series_v112):
    p = pylbo.plot_spectrum_multi(series_v112, xdata="k3")
    assert np.allclose(series_v112.parameters["k3"], p.xdata)


def test_multispectrum_plot_xdata_list(series_v112):
    p = pylbo.plot_spectrum_multi(series_v112, xdata=[1, 2, 3])
    p.draw()
    assert np.allclose([1, 2, 3], p.xdata)
    xlims = p.ax.get_xlim()
    assert xlims[0] <= 1
    assert xlims[1] >= 3


def test_multispectrum_plot_ydata_real(ds_v112, series_v112):
    p = pylbo.plot_spectrum_multi(series_v112, xdata="k2", use_real_parts=True)
    evs = ds_v112.eigenvalues.real
    for data in p.ydata:
        assert not np.all(np.isnan(data))
        assert not np.all(np.isnan(evs))
        mask = np.invert(np.isnan(data) | np.isnan(evs))
        assert np.allclose(data[mask], evs[mask])


def test_multispectrum_plot_ydata_imag(ds_v112, series_v112):
    p = pylbo.plot_spectrum_multi(series_v112, xdata="k2", use_real_parts=False)
    evs = ds_v112.eigenvalues.imag
    for data in p.ydata:
        assert not np.all(np.isnan(data))
        assert not np.all(np.isnan(evs))
        mask = np.invert(np.isnan(data) | np.isnan(evs))
        assert np.allclose(data[mask], evs[mask])


def test_multispectrum_plot_ydata_squared(ds_v112, series_v112):
    p = pylbo.plot_spectrum_multi(
        series_v112, xdata="k3", use_squared_omega=True, use_real_parts=True
    )
    evs = (ds_v112.eigenvalues**2).real
    for data in p.ydata:
        assert not np.all(np.isnan(data))
        assert not np.all(np.isnan(evs))
        mask = np.invert(np.isnan(data) | np.isnan(evs))
        assert np.allclose(data[mask], evs[mask])


def test_multispectrum_plot_set_scaling(series_v112):
    p = pylbo.plot_spectrum_multi(
        series_v112,
        xdata=np.arange(0, len(series_v112)),
        use_squared_omega=False,
        use_real_parts=True,
    )
    p.draw()
    ylims = np.array(p.ax.get_ylim(), dtype=float)
    xscale = 5.5
    yscale = 2.4
    p.set_x_scaling(xscale)
    p.set_y_scaling(yscale)
    assert np.allclose(ylims * yscale, p.ax.get_ylim())
    assert np.allclose(p.x_scaling, [xscale] * len(series_v112))
    assert np.allclose(p.y_scaling, [yscale] * len(series_v112))


def test_multispectrum_plot_continua(series_v112):
    p = pylbo.plot_spectrum_multi(series_v112, xdata=np.arange(0, len(series_v112)))
    p.add_continua()
    labels = p.ax.get_legend_handles_labels()[1]
    assert len(labels) > 2
    assert getattr(p, "c_handler", None) is not None


def test_multispectrum_plot_continua_squared(series_v112):
    p = pylbo.plot_spectrum_multi(series_v112, xdata="k3", use_squared_omega=True)
    p.add_continua()
    labels = p.ax.get_legend_handles_labels()[1]
    assert len(labels) == 3  # alfven+, slow+, doppler
    assert getattr(p, "c_handler", None) is not None


def test_multispectrum_plot_eigenfunctions(series_v112):
    p = pylbo.plot_spectrum_multi(series_v112, xdata=np.arange(0, len(series_v112)))
    p.add_eigenfunctions()
    assert p.ax.get_subplotspec().get_geometry()[0:2] == (1, 2)
    assert getattr(p, "ef_handler", None) is not None
    assert getattr(p, "ef_ax", None) is not None


def test_multispectrum_plot_eigenfunctions_notpresent(series_v100):
    p = pylbo.plot_spectrum_multi(series_v100, xdata="k2")
    with pytest.raises(EigenfunctionsNotPresent):
        p.add_eigenfunctions()


def test_multispectrum_plot_nobg_continua(series_v200_nobg):
    p = pylbo.plot_spectrum_multi(series_v200_nobg, xdata="k3")
    p.add_continua()
    assert getattr(p, "c_handler") is None


def test_merged_plot(series_v112):
    p = pylbo.plot_merged_spectrum(series_v112)
    p.draw()
    labels = p.ax.get_legend_handles_labels()[1]
    assert len(labels) == len(series_v112)


def test_merged_plot_legend_present(series_v112):
    p = pylbo.plot_merged_spectrum(series_v112)
    p.draw()
    legend = [
        child for child in p.ax.get_children() if isinstance(child, mpl.legend.Legend)
    ]
    assert legend


def test_merged_plot_single_color(series_v112):
    p = pylbo.plot_merged_spectrum(series_v112, color="blue")
    p.draw()
    legend = [
        child for child in p.ax.get_children() if isinstance(child, mpl.legend.Legend)
    ]
    assert not legend


def test_merged_plot_eigenfunctions(series_v112):
    p = pylbo.plot_merged_spectrum(series_v112)
    p.draw()
    p.add_eigenfunctions()
    assert p.ax.get_subplotspec().get_geometry()[0:2] == (1, 2)
    assert getattr(p, "ef_handler", None) is not None
    assert getattr(p, "ef_ax", None) is not None


def test_comparison_plot(ds_v100, ds_v112):
    p = pylbo.plot_spectrum_comparison(ds_v100, ds_v112)
    p.draw()
    assert all(isinstance(panel, SingleSpectrumPlot) for panel in (p.panel1, p.panel2))


def test_comparison_plot_geometry(ds_v100, ds_v112):
    p = pylbo.plot_spectrum_comparison(ds_v100, ds_v112)
    p.draw()
    assert p.panel1.ax.get_subplotspec().get_geometry()[0:3] == (1, 2, 0)
    assert p.panel2.ax.get_subplotspec().get_geometry()[0:3] == (1, 2, 1)


def test_comparison_plot_locked_view(ds_v100, ds_v112):
    p = pylbo.plot_spectrum_comparison(ds_v100, ds_v112, lock_zoom=True)
    p.draw()
    ax1 = p.panel1.ax
    ax2 = p.panel2.ax
    assert np.allclose(ax1.get_xlim(), ax2.get_xlim())
    assert np.allclose(ax1.get_ylim(), ax2.get_ylim())
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-25, 25)
    assert np.allclose(ax1.get_xlim(), ax2.get_xlim())
    assert np.allclose(ax1.get_ylim(), ax2.get_ylim())


def test_comparison_plot_set_x_scaling(ds_v112, ds_v112_eta):
    p = pylbo.plot_spectrum_comparison(ds_v112, ds_v112_eta)
    p.draw()
    xscale = 3.2
    xlims1 = np.array(p.panel1.ax.get_xlim(), dtype=float)
    xlims2 = np.array(p.panel2.ax.get_xlim(), dtype=float)
    p.set_x_scaling(xscale)
    assert np.allclose(xscale * xlims1, p.panel1.ax.get_xlim())
    assert np.allclose(xscale * xlims2, p.panel2.ax.get_xlim())


def test_comparison_plot_set_y_scaling(ds_v112, ds_v112_eta):
    p = pylbo.plot_spectrum_comparison(ds_v112, ds_v112_eta)
    p.draw()
    yscale = -7.1
    ylims1 = np.flip(np.array(p.panel1.ax.get_ylim(), dtype=float))
    ylims2 = np.flip(np.array(p.panel2.ax.get_ylim(), dtype=float))
    p.set_y_scaling(yscale)
    assert np.allclose(yscale * ylims1, p.panel1.ax.get_ylim())
    assert np.allclose(yscale * ylims2, p.panel2.ax.get_ylim())


def test_comparison_plot_continua(ds_v100, ds_v112):
    p = pylbo.plot_spectrum_comparison(ds_v100, ds_v112)
    p.draw()
    p.add_continua()
    for panel in (p.panel1, p.panel2):
        labels = panel.ax.get_legend_handles_labels()[1]
        assert len(labels) > 2


def test_comparison_plot_eigenfunctions(ds_v112):
    p = pylbo.plot_spectrum_comparison(ds_v112, ds_v112)
    p.draw()
    p.add_eigenfunctions()
    assert get_axis_geometry(p.panel1.ax) == (2, 1, 0)
    assert get_axis_geometry(p.panel1.ef_ax) == (2, 1, 1)
    assert get_axis_geometry(p.panel2.ax) == (2, 1, 0)
    assert get_axis_geometry(p.panel2.ef_ax) == (2, 1, 1)
    assert len(p.fig.get_axes()) == 4


def test_2d_visualisation_no_bg(ds_v200_tear_nobg):
    with pytest.raises(BackgroundNotPresent):
        pylbo.plot_2d_slice(
            ds_v200_tear_nobg,
            omega=0.1,
            ef_name="rho",
            u2=np.linspace(0, 1, 20),
            u3=0.5,
            time=0,
            slicing_axis="z",
            add_background=True,
        )
