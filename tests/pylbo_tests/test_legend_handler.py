import matplotlib as mpl
import pylbo
from pylbo.testing import MockPickEvent
from pylbo.visualisation.legend_handler import _get_legend_handles


def _create_profile_plot(ds):
    return pylbo.plot_equilibrium(ds)


def get_legend_artists(ax):
    return _get_legend_handles(ax.get_legend())


def get_children(ax, artists=None):
    artists = artists or get_legend_artists(ax)
    artist_labels = [artist.get_label() for artist in artists]
    return [child for child in ax.get_children() if child.get_label() in artist_labels]


def get_artist_from_label(ax, label):
    artists = get_legend_artists(ax)
    for artist in artists:
        if label in artist.get_label():
            return artist
    return None


def get_child_from_label(ax, label, artists=None):
    for child in get_children(ax, artists):
        if label in child.get_label():
            return child
    return None


def get_nb_visible_children(ax, artists=None):
    return [child.get_visible() for child in get_children(ax, artists)].count(True)


def _create_mockpickevent(p, ax=None):
    axis = p.ax if ax is None else ax
    return MockPickEvent(
        mouse_x=None, mouse_y=None, button=1, ds=p.data, axes=axis, figure=p.fig
    )


def _create_single_pickevent(p, artist, ax=None):
    event = _create_mockpickevent(p, ax)
    event.artist = artist
    p.leg_handle.on_legend_pick(event)


def test_legend_profile_nothing_clicked(ds_v200_mri_efs):
    p = _create_profile_plot(ds_v200_mri_efs)
    for child in get_children(p.ax):
        assert not child.get_visible()


def test_legend_profile_left_click(ds_v200_mri_efs):
    p = _create_profile_plot(ds_v200_mri_efs)
    label = "rho_0"
    _create_single_pickevent(p, artist=get_artist_from_label(p.ax, label))
    for child in get_children(p.ax):
        assert isinstance(child, mpl.lines.Line2D)
        if label in child.get_label():
            assert child.get_visible()
        else:
            assert not child.get_visible()
    assert get_nb_visible_children(p.ax) == 1


def test_legend_profile_multiple_clicked(ds_v200_mri_efs):
    p = _create_profile_plot(ds_v200_mri_efs)
    labels = ["rho_0", "T_0", "B_{02}"]
    for label in labels:
        _create_single_pickevent(p, artist=get_artist_from_label(p.ax, label))
    for child in get_children(p.ax):
        assert isinstance(child, mpl.lines.Line2D)
        if any(label in child.get_label() for label in labels):
            assert child.get_visible()
        else:
            assert not child.get_visible()
    assert get_nb_visible_children(p.ax) == 3


def test_legend_profile_clicked_toggle_off(ds_v200_mri_efs):
    p = _create_profile_plot(ds_v200_mri_efs)
    labels = ["rho_0", "T_0", "B_{02}"]
    for label in labels:
        _create_single_pickevent(p, artist=get_artist_from_label(p.ax, label))
    _create_single_pickevent(p, artist=get_artist_from_label(p.ax, labels[1]))
    assert get_nb_visible_children(p.ax) == 2
    assert get_child_from_label(p.ax, labels[1]).get_visible() is False


def test_legend_profile_clicked_derivative(ds_v200_mri_efs):
    p = _create_profile_plot(ds_v200_mri_efs)
    artists = [
        artist for artist in get_legend_artists(p.ax) if "partial" in artist.get_label()
    ]
    _create_single_pickevent(p, artist=artists[0], ax=p.ax2)
    assert get_nb_visible_children(p.ax) == 0
    assert get_nb_visible_children(p.ax2, artists) == 1
    assert (
        get_child_from_label(p.ax2, artists[0].get_label(), artists).get_visible()
        is True
    )
