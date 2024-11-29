# Copyright (c) 2023 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.
# -*- coding: utf-8 -*-
"""
Functions to make plotting better
Factored out from helper
@author: lloopik
Thanks Leon, for writing this!
@author2: MZwalua, added some features

Changelist
Version 1.0.1; 2023-11-10
 - Updated to work with matplotlib 3.8 (https://github.com/matplotlib/matplotlib/pull/26189)
Version 1.0.0; 2023-10-23
 - Added changelist and version numbers
 - Merged with pybms version

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.legend as mlegend
import numpy as np
import functools
import itertools
import operator
import io
from collections import defaultdict

try:
    from matplotlib.backend_bases import MouseButton
except ImportError:
    from enum import IntEnum

    class MouseButton(IntEnum):
        LEFT = 1
        MIDDLE = 2
        RIGHT = 3
        BACK = 8
        FORWARD = 9


try:
    from PyQt5 import QtCore
    from PyQt5.Qt import QApplication, QImage

    using_QT = True
except ImportError:
    using_QT = False
try:
    import pandas as pd

    using_pandas = True
except ImportError:
    using_pandas = False
try:
    import numba

    njit = numba.njit
    using_numba = True
except ImportError:
    njit = lambda x: x
    using_numba = False


sleep = plt.pause


def init():
    """
    Initialize plthelper:
        - Hook the Axes constructor
        - Use the ggplot style
        - Turn off the all_axes keymap
    """
    hook()
    plt.style.use("ggplot")
    # Use ugly workaround to prevent a deprecationwarning, see
    # https://github.com/matplotlib/matplotlib/issues/18979
    dict.__setitem__(mpl.rcParams, "keymap.all_axes", [])
    # rcParams['keymap.all_axes'] = None


def wrap(oldFunc, before=None, after=None, overWrite=True, wid="ll_wrapped"):
    """
    Wrap/monkeypatch a function
    Returns the wrapped function.
    """
    if hasattr(oldFunc, wid):
        if overWrite:
            oldFunc = getattr(oldFunc, wid)
        else:
            return oldFunc

    @functools.wraps(oldFunc)
    def newFunc(*args, **kwargs):
        if before is not None:
            bret = before(*args, **kwargs)
            if bret is not None:
                (args, kwargs) = bret

        ret = oldFunc(*args, **kwargs)

        if after is not None:
            ret = after(ret, *args, **kwargs)
        return ret

    setattr(newFunc, wid, oldFunc)
    return newFunc


def unwrap(func, wid="ll_wrapped"):
    """
    Unwrap a function
    Returns the original function
    """
    if hasattr(func, wid):
        func = getattr(func, wid)
    return func


def hook():
    """
    Hook the Axes constructor to enable all plot enhancements
    """

    def doEnable(ret, self, *args, **kwargs):
        if self is not None:
            enable(self)
        return ret

    plt.Axes.__init__ = wrap(plt.Axes.__init__, after=doEnable)


def unhook():
    """
    Unook the Axes constructor to disable all plot enhancements
    """
    plt.Axes.__init__ = unwrap(plt.Axes.__init__)


def enable(ax=None):
    """
    Enable all plot enhancements
    """
    if ax is None:
        ax = plt.gca()
    ZoomControl(ax)
    ToggleControl(ax)
    CopyControl(ax.figure)
    MarkerControl(ax)

    ax.engFormat = functools.partial(engFormat, ax)


def disable(ax=None):
    if ax is None:
        ax = plt.gca()
    ZoomControl.remove(ax)
    ToggleControl.remove(ax)
    CopyControl.remove(ax.figure)
    MarkerControl.remove(ax)

    if hasattr(ax, "engFormat"):
        del ax.engFormat


# %% Constants
UNITS_NORM = {
    "Ohm": "Ω",
    "degree C": "°C",
    "deg C": "°C",
    "degree": "°",
}

UNITS_DIV = {
    ("V", "A"): "Ω",
    ("V", "Ω"): "A",
    ("A", "V"): "S",
    ("A", "S"): "V",
    ("1", "Ω"): "S",
    ("1", "S"): "Ω",
}

SCALAR_FORMATTER = ticker.ScalarFormatter()


# %% Save and restore plots
def _fixClass(obj):
    """
    Needed if the file has been modified, and things have been auto-reloaded...
    """
    cls = obj.__class__
    mod = cls.__module__
    if mod != __name__:
        return
    import sys
    from pickle import _getattribute

    __import__(mod, level=0)
    module = sys.modules[mod]
    try:
        cls2, _ = _getattribute(module, cls.__name__)
    except Exception:
        return
    obj.__class__ = cls2


def _fixFigure(fig):
    """
    Update old references
    """
    for ax in fig.axes:
        for axis in [ax.xaxis, ax.yaxis]:
            for ticks in [axis.major, axis.minor]:
                _fixClass(ticks.formatter)


def _saveFig(zfile, iname, fig):
    import pickle

    if fig.canvas is None:
        raise Exception(
            "Cannot find canvas. Try resurecting the figure first."
        )
    for ax in fig.axes:
        disable(ax)
    with zfile.open(iname, "w", force_zip64=True) as f:
        pickle.dump(fig, f)
    for ax in fig.axes:
        enable(ax)


def _createFigZip(filename, force, callback):
    import zipfile
    from pathlib import Path
    from numpy.lib.npyio import zipfile_factory

    if Path(filename).exists() and not force:
        raise Exception("File already exists!")
    with zipfile_factory(
        filename, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as zfile:
        callback(zfile)


def saveFig(fig, filename, force=False):
    if not filename.endswith(".fig"):
        filename += ".fig"

    def save(zfile):
        _saveFig(zfile, "fig", fig)

    _createFigZip(filename, force, save)


def saveAllFigs(filename, force=False):
    if not filename.endswith(".figs"):
        filename += ".figs"

    def save(zfile):
        for num in plt.get_fignums():
            iname = f"{num}.fig"
            fig = plt.figure(num)
            _saveFig(zfile, iname, fig)

    _createFigZip(filename, force, save)


def _loadFig(zfile, iname):
    import pickle

    with zfile.open(iname, "r", force_zip64=True) as f:
        fig = pickle.load(f)
    for ax in fig.axes:
        enable(ax)
    return fig


def loadFig(filename):
    from numpy.lib.npyio import zipfile_factory

    with zipfile_factory(filename, mode="r") as zfile:
        if len(zfile.namelist()) == 1 and zfile.namelist()[0] == "fig":
            figs = _loadFig(zfile, "fig")
        else:
            figs = []
            for file in zfile.namelist():
                figs.append(_loadFig(zfile, file))
    return figs


def resurect(fig):
    dummy = plt.figure()
    manager = dummy.canvas.manager
    manager.canvas.figure = fig
    fig.set_canvas(manager.canvas)

    for ax in fig.axes:
        enable(ax)


# %% Helper Functions
def subplots(name=None, *args, **kwargs):
    kwargs["num"] = name
    if name in plt.get_figlabels() or (
        isinstance(name, int) and name in plt.get_fignums()
    ):
        fig = plt.figure(num=name)
        if hasattr(
            fig, "_plthelper_subplots_args"
        ) and fig._plthelper_subplots_args == (args, kwargs):
            return fig._plthelper_subplots_res
    res = plt.subplots(*args, **kwargs)
    res[0]._plthelper_subplots_args = (args, kwargs)
    res[0]._plthelper_subplots_res = res
    return res


def engFormat(
    ax=None,
    axis="y",
    unit="",
    places=None,
    sep=" ",
    useOffset=False,
    *,
    norm_unit=True,
):
    """
    Set the engineering formatter to one of the axis
    """
    if ax is None:
        ax = plt.gca()
    if norm_unit:
        unit = UNITS_NORM.get(unit, unit)
    if axis == "y":
        axis = ax.yaxis
    elif axis == "x":
        axis = ax.xaxis
    else:
        raise ValueError("axis should be 'x' or  'y'")
    if unit != "time":
        form = EngFormatter(unit, places, sep)
    else:
        form = TimeFormatter("s", places, sep)
        axis.set_major_locator(TimeLocator())
    form.useOffset = useOffset
    axis.set_major_formatter(form)

    if axis.get_scale() == "log":
        try:
            base = axis.get_major_locator()._base
        except AttributeError:
            base = 10
        minform = LogEngMixin.fromFormatter(
            ticker.LogFormatter(base), unit, places, sep
        )
        axis.set_minor_formatter(minform)


def fixtwinx(ax1, ax2, rescale=True):
    """
    Freeze the twinx relationship,
    when rescale is true, it will also rescale ax2 to a nice interval
    """
    if rescale:
        niceSteps = np.array([0.8, 1, 2, 2.5, 3, 5, 8, 10, 20])
        interval = np.diff(ax2.yaxis.get_data_interval())[0]
        N = len(ax1.get_yticks())
        step = np.ceil(interval / (N - 1))
        tens = np.power(10, np.floor(np.log10(step)))
        possibleSteps = niceSteps[(niceSteps / step * tens > 1)] * tens

        step = possibleSteps[0]
        otherticks = ax1.get_yticks()

        # scaling to bottom now, improvement would be to scale to middle
        scale = step / (otherticks[1] - otherticks[0])
        bottom = (
            otherticks[0] - ax1.yaxis.get_data_interval()[0]
        ) * scale + ax2.yaxis.get_data_interval()[0]
        bottom = np.floor(bottom / step) * step

        ticks = np.arange(bottom, bottom + N * step, step)

        ax2.set_yticks(ticks)
        ax2.set_ylim((ax1.get_ylim() - otherticks[0]) * scale + ticks[0])
    ax2.yaxis.set_major_locator(FollowLocator(ax2, ax1))
    ax2.grid(None)


def swaplines(ax, l1, l2):
    tmp = ax.lines[l1]
    ax.lines[l1] = ax.lines[l2]
    ax.lines[l2] = tmp

    tmpcol = ax.lines[l1].get_color()
    ax.lines[l1].set_color(ax.lines[l2].get_color())
    ax.lines[l2].set_color(tmpcol)

    ax.legend()
    ax.figure.canvas.draw()


def saveToClipboard(figure):
    """
    Copy a figureto the clipboard
    """
    if using_QT:
        buf = io.BytesIO()
        figure.savefig(buf, dpi=5 * figure.dpi)
        QApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
        buf.close()
    else:
        print("Not implemented for your setup!")


if using_QT:
    # From:
    # https://stackoverflow.com/questions/19662906/plotting-with-matplotlib-in-threads
    class Call_in_QT_main_loop(QtCore.QObject):
        signal = QtCore.pyqtSignal()

        def __init__(self, func):
            super().__init__()
            self.func = func
            self.args = list()
            self.kwargs = dict()
            self.signal.connect(self._target)

        def _target(self):
            self.func(*self.args, **self.kwargs)

        def __call__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.signal.emit()


def getMap(center, alpha=1):
    if not center.startswith("#"):
        center = colors.get_named_colors_mapping()[center]
    r, g, b = colors.to_rgb(center)
    segmentdata = {
        "red": [(0, 0, r * 0.9), (1, min(r * 1.1, 1), 1)],
        "green": [(0, 0, g * 0.9), (1, min(g * 1.1, 1), 1)],
        "blue": [(0, 0, b * 0.9), (1, min(b * 1.1, 1), 1)],
        "alpha": [(0, 0, alpha), (1, alpha, 1)],
    }
    return colors.LinearSegmentedColormap(center + "_map", segmentdata)


@njit
def _moving_avg(xdata, ydata, hint):
    movdata = np.zeros_like(ydata)
    for i in range(len(xdata)):
        filt = (xdata > xdata[i] - hint) & (xdata < xdata[i] + hint)
        movdata[i] = np.mean(ydata[filt])
    return movdata


def moving_avg(line, smooth=0.1, rms=False, db=False):
    "Do not use, not finished"
    ax = line.axes
    xdata, ydata = line.get_data(True)
    if ax.xaxis.get_scale() == "log":
        xdata = np.log(xdata)

    if db:
        ydata = 10 ** (ydata / 20)

    if rms:
        ydata = ydata**2

    hint = (max(xdata) - min(xdata)) * smooth / 2
    movdata = _moving_avg(xdata, ydata, hint)

    if rms:
        movdata = np.sqrt(movdata)

    if db:
        movdata = 20 * np.log10(movdata)

    return movdata


if using_pandas:

    def plotdf(
        d: pd.DataFrame, x="temp", ax=None, xd=None, groupby=[], **kwargs
    ):
        """
        Plot multiindexed dataframe
        x: what to put on the x axis
        ax: axis to use
        xd: x data (for x vs y plots)
        groupby: what to group by.
        """
        if x not in d.index.names and xd is None:
            raise Exception("Don't know what to put on x axis")
        for k, v in kwargs.items():
            d = d.xs(v, level=k, drop_level=not (isinstance(v, slice)))
            if xd is not None:
                xd = xd.xs(v, level=k, drop_level=not (isinstance(v, slice)))
        unstack = [
            name for name in d.index.names if name != x and name not in groupby
        ]
        if ax is None:
            ax = plt.gca()
        d = d.unstack(unstack)
        if xd is None and groupby == []:
            d.plot(ax=ax)
        elif xd is not None and groupby == []:
            xd = xd.unstack(unstack)

            for c in d.columns:
                ax.plot(xd[c], d[c], label=c)
            #            if len(unstack) == 1:
            #                ax.legend(title=unstack[0])
            #            else:
            #                ax.legend(title=f'({", ".join(unstack)})')

            labels = d.columns.names
            if len(labels) == 1:
                ax.legend(title=labels[0])
            else:
                ax.legend(title=f'({", ".join(labels)})')
        elif xd is None and groupby != []:
            g = d.groupby(groupby)
            for key, group in g:
                nextcolor = next(ax._get_lines.prop_cycler)["color"]
                group.unstack(groupby).plot(ax=ax, cmap=getMap(nextcolor))
        elif xd is not None and groupby != []:
            g = d.groupby(groupby)
            xd = xd.unstack(unstack)
            xg = xd.groupby(groupby)
            for key, group in g:
                xgroup = xg.get_group(key)
                nextcolor = next(ax._get_lines.prop_cycler)["color"]
                group = group.unstack(groupby)
                xgroup = xgroup.unstack(groupby)
                cmap = getMap(nextcolor)
                N = len(d.columns)
                cs = np.linspace(0, 256, N)
                for c, ci in zip(d.columns, cs):
                    ax.plot(xgroup[c], group[c], color=cmap(ci), label=c)
            labels = d.columns.names
            if len(labels) == 1:
                ax.legend(title=labels[0])
            else:
                ax.legend(title=f'({", ".join(labels)})')
            # raise NotImplementedError

    def plotdf2(
        x,
        y,
        data,
        ax=None,
        hue=None,
        shape=None,
        fit_order=False,
        label_coefs=False,
        linestyle="",
    ):
        "Similar to above, different uses"
        if ax is None:
            _, ax = plt.subplots()

        x = data[x]
        y = data[y]
        fitx = np.linspace(x.min(), x.max(), 1000)

        # https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
        all_shapes = itertools.cycle([".", "*", "p", "s", "v", "1", "^", "x"])
        shapes = defaultdict(lambda: next(all_shapes))
        colors = defaultdict(lambda: None)
        shapes_lines = defaultdict(list)
        colors_lines = defaultdict(list)
        fit_lines = []

        legtitle = []
        keys = []
        if hue is not None:
            legtitle.append(hue)
            keys.append(data[hue])
        if shape is not None:
            legtitle.append(shape)
            keys.append(data[shape])
        done = set()
        mappings = []
        for key in itertools.product(*keys):
            if key in done:
                continue
            done.add(key)
            filt = functools.reduce(
                operator.and_,
                itertools.starmap(operator.eq, (zip(key, keys))),
                x == x,
            )
            if not any(filt):
                continue
            fx = x[filt]
            fy = y[filt]

            ikey = iter(key)
            color_key = None if hue is None else next(ikey)
            marker_key = None if shape is None else next(ikey)

            color = None if hue is None else colors[color_key]
            marker = "." if shape is None else shapes[marker_key]

            # label = key if len(key) > 1 else key[0]
            (sc,) = ax.plot(
                fx,
                fy,
                linestyle=linestyle,
                marker=marker,
                color=color,  # , label=label
            )
            color = sc.get_color()
            colors_lines[color_key].append(sc)
            shapes_lines[marker_key].append(sc)

            ikey = iter(key)
            if hue is not None:
                colors[next(ikey)] = color
            if fit_order:
                fit = np.polynomial.Polynomial.fit(fx, fy, fit_order)

                def power(i):
                    if i == 0:
                        return ""
                    elif i == 1:
                        return "*x"
                    else:
                        return f"*x^{i}"

                if label_coefs:
                    fitlabel = " + ".join(
                        f"{a:.4g}{power(i)}"
                        for i, a in enumerate(fit.convert().coef)
                    )
                    (line,) = ax.plot(
                        fitx, fit(fitx), color=color, label=fitlabel
                    )
                    mappings.append([line])
                else:
                    (line,) = ax.plot(fitx, fit(fitx), color=color)
                    colors_lines[color_key].append(line)
                    shapes_lines[marker_key].append(line)
                    fit_lines.append(line)

        # dummy plots for legend entries
        if hue is not None:
            for key, color in colors.items():
                (l,) = ax.plot(
                    [],
                    [],
                    marker="s",
                    color=color,
                    linestyle="",
                    label=f"{hue}: {key}",
                )
                mappings.append([l, *colors_lines[key]])
        if shape is not None:
            for key, marker in shapes.items():
                (l,) = ax.plot(
                    [],
                    [],
                    marker=marker,
                    color="k",
                    linestyle="",
                    label=f"{shape}: {key}",
                )
                mappings.append([l, *shapes_lines[key]])
        if fit_order:
            (l,) = ax.plot(
                [], [], marker="", color="k", label=f"fit (order={fit_order})"
            )
            mappings.append([l, *fit_lines])

        leg = ax.legend()
        leg.ll_mapping = mappings
        if hasattr(ax, "ll_toggle_ctrl"):
            ax.ll_toggle_ctrl.updateDict()


# %% Helper Classes
class Unit:
    def __new__(cls, unit):
        unit = cls._UNITS.get(str(unit))
        if unit is None:
            return super().__new__(cls)
        else:
            return unit

    def __init__(self, unit):
        print(unit)


# Inspired by https://stackoverflow.com/a/58324984
# by Ian Hicks
class Custom_labels:
    def __init__(self):
        # labels[ax] is a list of tuples (object, label, mapping)
        #  mapping is a list of all things to make invisible when toggling, or
        #  None to just toggle the object.
        self.labels = defaultdict(list)

    def add_violin(self, violin, label):
        bodies = violin["bodies"]
        cbars = violin["cbars"]
        ax = bodies[0].axes
        mapping = [cbars, *bodies, violin["cmaxes"], violin["cmins"]]
        self.labels[ax].append((cbars, label, mapping))

    def add_lines(self, lines, label):
        ax = lines[0].axes
        self.labels[ax].append((lines[0], label, lines))

    def legend(self, *args, **kwargs):
        for ax in self.labels:
            objects, labels, mappings = zip(*self.labels[ax])
            leg = ax.legend(objects, labels, *args, **kwargs)
            leg.ll_mapping = mappings
            if hasattr(ax, "ll_toggle_ctrl"):
                ax.ll_toggle_ctrl.updateDict()


class EngFormatter(ticker.EngFormatter):
    """
    The ticker.EngFormatter with offset
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._useOffset = False
        self._offset_threshold = 2
        self.offset = 0

    def __call__(self, x, pos=None, no_offset=False):
        if no_offset:
            xp = x
        elif np.nextafter(x, self.offset) == self.offset:
            xp = 0
        else:
            xp = x - self.offset
        return self.do_format(xp, pos)

    def do_format(self, x, pos):
        return super().__call__(x, pos)

    @property
    def useOffset(self):
        return self._useOffset

    @useOffset.setter
    def useOffset(self, val):
        if val in [True, False]:
            self.offset = 0
            self._useOffset = val
        else:
            self._useOffset = False
            self.offset = val

    def get_offset(self):
        """
        Return scientific notation, plus offset.
        """
        if len(self.locs) == 0:
            return ""
        offsetStr = ""
        if self.offset:
            offsetStr = self.__call__(self.offset, no_offset=True)
            if self.offset > 0:
                offsetStr = "+" + offsetStr
        return self.fix_minus(offsetStr)

    def set_locs(self, locs):
        """
        Set the locations of the ticks.
        """
        self.locs = locs
        if len(self.locs) > 0:
            if self._useOffset:
                self._compute_offset()

    def _compute_offset(self):
        locs = self.locs
        if locs is None or not len(locs):
            self.offset = 0
            return
        # Restrict to visible ticks.
        vmin, vmax = sorted(self.axis.get_view_interval())
        locs = np.asarray(locs)
        locs = locs[(vmin <= locs) & (locs <= vmax)]
        if not len(locs):
            self.offset = 0
            return
        lmin, lmax = locs.min(), locs.max()
        # Only use offset if there are at least two ticks and every tick has
        # the same sign.
        if lmin == lmax or lmin <= 0 <= lmax:
            self.offset = 0
            return
        # min, max comparing absolute values (we want division to round towards
        # zero so we work on absolute values).
        abs_min, abs_max = sorted([abs(float(lmin)), abs(float(lmax))])
        sign = np.copysign(1, lmin)
        # What is the smallest power of ten such that abs_min and abs_max are
        # equal up to that precision?
        # Note: Internally using oom instead of 10 ** oom avoids some numerical
        # accuracy issues.
        oom_max = np.ceil(np.log10(abs_max))
        oom = 1 + next(
            oom
            for oom in itertools.count(oom_max, -1)
            if abs_min // 10**oom != abs_max // 10**oom
        )
        if (abs_max - abs_min) / 10**oom <= 1e-2:
            # Handle the case of straddling a multiple of a large power of ten
            # (relative to the span).
            # What is the smallest power of ten such that abs_min and abs_max
            # are no more than 1 apart at that precision?
            oom = 1 + next(
                oom
                for oom in itertools.count(oom_max, -1)
                if abs_max // 10**oom - abs_min // 10**oom > 1
            )
        # Only use offset if it saves at least _offset_threshold digits.
        n = self._offset_threshold - 1
        if abs_max // 10**oom >= 10**n:
            os = sign * (abs_max // 10**oom) * 10**oom
            self.offset = os  # np.nextafter(os, 0)
        else:
            self.offset = 0


class EngMixin:
    _BaseEngFormatter = EngFormatter
    __subclasses = {}

    def __new__(self, *args, **kwargs):
        raise Exception("Cannot instantiate EngMixin")

    @classmethod
    def _restore(cls, oldcls):
        return cls.fromFormatter(oldcls())

    def __reduce__(self):
        return (EngMixin._restore, (self.__oldcls,), self.__dict__)

    @classmethod
    def fromFormatter(
        cls,
        oldFormatter,
        unit="",
        places=None,
        sep=" ",
        *,
        usetex=None,
        useMathText=None,
    ):
        oldcls = oldFormatter.__class__
        newcls = cls.__subclasses.get((cls, oldcls))
        if newcls is None:
            newcls = type(
                oldcls.__name__ + "_" + cls.__name__,
                (cls, oldcls, cls._BaseEngFormatter),
                {},
            )
            cls.__subclasses[(cls, oldcls)] = newcls
        self = oldFormatter
        self.__class__ = newcls
        self.__oldcls = oldcls
        cls._BaseEngFormatter.__init__(
            self, unit, places, sep, usetex=usetex, useMathText=useMathText
        )
        return self


class LogEngMixin(EngMixin):
    def _num_to_string(self, x, vmin, vmax):
        return self._BaseEngFormatter.__call__(self, x)


class TimeFormatter(EngFormatter):
    """
    A formatter that nicely prints times
    """

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            assert args[0] == "s", "Units should be seconds"
            super().__init__(*args, **kwargs)
        else:
            super().__init__(unit="s", **kwargs)

    def do_format(self, x, pos=None):
        if abs(x) < 60:
            return super().do_format(x, pos)
        s = self.format_time(x)
        return self.fix_minus(s)

    @staticmethod
    def format_time(secs):
        s = "-" if secs < 0 else ""
        (secs, subsecs) = divmod(abs(secs), 1)
        (mins, secs) = divmod(secs, 60)
        (hours, mins) = divmod(mins, 60)
        subsecs = int(10 * subsecs + 0.5)

        # result
        res = s
        # temporary result, only add to final result if something non-zero follows
        tres = ""
        unit = None
        for t, u in [(hours, "h"), (mins, "m"), (secs, "s"), (subsecs, "ss")]:
            if unit is None and t == 0 and u != "s":
                continue
            if unit is None:
                unit = u
                res += "{:d}".format(int(t))
            elif u != "ss":
                # subseconds need a .
                tres += ":{:0>2d}".format(int(t))
                if t != 0:
                    res += tres
                    tres = ""
            elif t != 0:
                # skip if no subseconds
                tres += ".{:1d}".format(t)
                res += tres

        return res + f" {unit}"


class FollowLocator(ticker.Locator):
    """
    A locator that follows another axis
    """

    def __init__(self, myAx, followAx):
        self._fax = followAx
        otherlim = self._fax.get_ylim()
        mylim = myAx.get_ylim()
        self._scale = (mylim[-1] - mylim[0]) / (otherlim[-1] - otherlim[0])
        self._offset = mylim[0] - otherlim[0] * self._scale
        # print(self._scale)
        # print(self._offset)

    def __call__(self):
        vmax, vmin = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        return self._fax.get_yticks() * self._scale + self._offset

    def view_limits(self, dmin, dmax):
        return np.array(self._fax.get_ylim()) * self._scale + self._offset


class TimeLocator(ticker.AutoLocator):
    """
    A locator that nicely locates times (in seconds)
    """

    def __init__(self):
        super().__init__()

    def _raw_ticks(self, vmin, vmax):
        """
        Generate a list of tick locations including the range *vmin* to
        *vmax*.  In some applications, one or both of the end locations
        will not be needed, in which case they are trimmed off
        elsewhere.
        """
        if self._nbins == "auto":
            if self.axis is not None:
                nbins = np.clip(
                    self.axis.get_tick_space(),
                    max(1, self._min_n_ticks - 1),
                    9,
                )
            else:
                nbins = 9
        else:
            nbins = self._nbins

        rescale = 3600
        vmin /= rescale
        vmax /= rescale
        scale, offset = ticker.scale_range(vmin, vmax, nbins)
        while scale < 1 and rescale > 1:
            rescale /= 60
            vmin *= 60
            vmax *= 60
            scale, offset = ticker.scale_range(vmin, vmax, nbins)
        _vmin = vmin - offset
        _vmax = vmax - offset
        raw_step = (_vmax - _vmin) / nbins
        steps = self._extended_steps * scale
        if self._integer:
            # For steps > 1, keep only integer values.
            igood = (steps < 1) | (np.abs(steps - np.round(steps)) < 0.001)
            steps = steps[igood]

        istep = np.nonzero(steps >= raw_step)[0][0]

        # This is an upper limit; move to smaller steps if necessary.
        for istep in reversed(range(istep + 1)):
            step = steps[istep]

            if (
                self._integer
                and np.floor(_vmax) - np.ceil(_vmin) >= self._min_n_ticks - 1
            ):
                step = max(1, step)
            best_vmin = (_vmin // step) * step

            # Find tick locations spanning the vmin-vmax range, taking into
            # account degradation of precision when there is a large offset.
            # The edge ticks beyond vmin and/or vmax are needed for the
            # "round_numbers" autolimit mode.
            edge = ticker._Edge_integer(step, offset)
            low = edge.le(_vmin - best_vmin)
            high = edge.ge(_vmax - best_vmin)
            ticks = np.arange(low, high + 1) * step + best_vmin
            # Count only the ticks that will be displayed.
            nticks = ((ticks <= _vmax) & (ticks >= _vmin)).sum()
            if nticks >= self._min_n_ticks:
                break
        return (ticks + offset) * rescale


# %% Control hook classes:
class CopyControl:
    """
    Hook ctrl+c to copy the figure
    """

    def __init__(self, fig):
        if hasattr(fig, "ll_copy_ctrl"):
            fig.ll_copy_ctrl.disconnect()
            del fig.ll_copy_ctrl
        self.fig = fig
        self.connect()
        fig.ll_copy_ctrl = self

    @classmethod
    def remove(cls, fig):
        if hasattr(fig, "ll_copy_ctrl"):
            fig.ll_copy_ctrl.disconnect()
            del fig.ll_copy_ctrl

    def connect(self):
        "Attach the callbacks"
        connect = self.fig.canvas.mpl_connect
        self.press_cid = connect("key_press_event", self.press_fun)

    def disconnect(self):
        "Remove the callbacks"
        disconnect = self.fig.canvas.mpl_disconnect
        disconnect(self.press_cid)

    def press_fun(self, event):
        if event.key == "ctrl+c":
            saveToClipboard(self.fig)
            print("Figure copied to clipboard")


class ZoomControl:
    """
    Make zooming and navigating a plot more confortable by:
        - Mouse scroll: zoom in/out
        - Middle mouse: autoscale
        - Right click: zoom to rectangle
        - Shift makes everyghing work only on the x axis
        - Ctrl makes everyghing work only on the y axis
        - 'm' Toggles moving
        - When moving is enabled, left mouse button moves around
    """

    def __init__(self, ax, base_scale=1.2):
        self.remove(ax)

        self.ax = ax
        self.base_scale = base_scale
        self.drawing = False
        self.moving = False
        self.movable = False

        self.connect()

        ax.ll_zoom_ctrl = self

    @classmethod
    def remove(cls, ax):
        if hasattr(ax, "ll_zoom_ctrl"):
            ax.ll_zoom_ctrl.disconnect()
            del ax.ll_zoom_ctrl

    def connect(self):
        "Attach the callbacks"
        connect = self.ax.figure.canvas.mpl_connect
        self.scroll_cid = connect("scroll_event", self.scroll_fun)
        self.press_cid = connect("button_press_event", self.press_fun)
        self.release_cid = connect("button_release_event", self.release_fun)
        self.key_cid = connect("key_press_event", self.key_fun)

    def connect_move(self):
        connect = self.ax.figure.canvas.mpl_connect
        self.move_cid = connect("motion_notify_event", self.move_fun)
        self.leave_cid = connect("axes_leave_event", self.move_fun)

    def disconnect(self):
        "Remove the callbacks"
        disconnect = self.ax.figure.canvas.mpl_disconnect
        disconnect(self.scroll_cid)
        disconnect(self.press_cid)
        disconnect(self.release_cid)
        disconnect(self.key_cid)
        if self.drawing:
            self.disconnect_move()
            self.rect.remove()
        if self.moving:
            self.disconnect_move()

    def disconnect_move(self):
        disconnect = self.ax.figure.canvas.mpl_disconnect
        disconnect(self.move_cid)
        disconnect(self.leave_cid)

    def redraw(self):
        canvas = self.ax.figure.canvas
        canvas.draw()
        if self.drawing:
            self.background = canvas.copy_from_bbox(self.ax.bbox)
            self.ax.draw_artist(self.rect)
            canvas.blit(self.ax.bbox)

    def to_log(self, xdata, ydata):
        if self.ax.get_xscale() == "log":
            xdata = np.log(xdata)
        if self.ax.get_yscale() == "log":
            ydata = np.log(ydata)
        return xdata, ydata

    def from_log(self, xdata, ydata):
        if self.ax.get_xscale() == "log":
            xdata = np.exp(xdata)
        if self.ax.get_yscale() == "log":
            ydata = np.exp(ydata)
        return xdata, ydata

    def key_fun(self, event):
        if event.key == "m":
            self.movable = not self.movable

    def scroll_fun(self, event):
        if event.inaxes != self.ax:
            return
        # get the current x and y limits
        (cur_xlim, cur_ylim) = self.to_log(
            self.ax.get_xlim(), self.ax.get_ylim()
        )

        # get event x and y location
        (xdata, ydata) = self.to_log(event.xdata, event.ydata)

        if event.button == "up":
            # deal with zoom in
            scale_factor = 1 / self.base_scale
        elif event.button == "down":
            # deal with zoom out
            scale_factor = self.base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        new_xlim = list(cur_xlim)
        new_ylim = list(cur_ylim)
        if event.key != "control":
            new_xlim[0] = xdata - (xdata - cur_xlim[0]) * scale_factor
            new_xlim[1] = xdata - (xdata - cur_xlim[1]) * scale_factor
        if event.key != "shift":
            new_ylim[0] = ydata - (ydata - cur_ylim[0]) * scale_factor
            new_ylim[1] = ydata - (ydata - cur_ylim[1]) * scale_factor
        (new_xlim, new_ylim) = self.from_log(new_xlim, new_ylim)

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)

        self.redraw()  # force re-draw

    def press_fun(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == MouseButton.MIDDLE:
            axis = "both"
            if event.key == "shift":
                axis = "x"
            elif event.key == "control":
                axis = "y"
            self.ax.relim(True)
            self.ax.autoscale(axis=axis)
            self.ax.figure.canvas.draw()
            return
        if event.button == MouseButton.RIGHT and not self.moving:
            self.start_draw(event)
        if (
            event.button == MouseButton.LEFT
            and self.movable
            and not self.drawing
        ):
            self.start_move(event)

    def release_fun(self, event):
        # only act when drawing
        if self.drawing and event.button == MouseButton.RIGHT:
            self.stop_draw(event)
        if self.moving and event.button == MouseButton.LEFT:
            self.stop_move(event)

    def move_fun(self, event):
        if event.inaxes != self.ax:
            return
        if self.drawing:
            self.move_draw(event)
        if self.moving:
            self.move_move(event)

    def start_draw(self, event):
        self.drawing = True
        self.startx = event.xdata
        self.starty = event.ydata

        self.connect_move()

        self.rect = self.ax.add_patch(
            patches.Rectangle(
                (self.startx, self.starty),
                0,
                0,
                fill=False,
                linewidth=1,
                edgecolor="black",
                zorder=10,
            )
        )

        self.rect.set_animated(True)
        self.redraw()

    def stop_draw(self, event):
        self.disconnect_move()

        self.rect.remove()

        self.drawing = False
        self.background = None

        if event.inaxes == self.ax:
            self.stopx = event.xdata  # get event x location
            self.stopy = event.ydata  # get event y location
        # print(f"Stopped drawing at {xdata} , {ydata}")

        if self.startx == self.stopx or self.starty == self.stopy:
            self.redraw()
            return
        ylim = self.ax.get_ylim()
        inverted_y = ylim[0] > ylim[1]
        xlim = self.ax.get_xlim()
        inverted_x = xlim[0] > xlim[1]

        self.ax.set_xlim(
            [min(self.startx, self.stopx), max(self.startx, self.stopx)]
        )
        self.ax.set_ylim(
            [min(self.starty, self.stopy), max(self.starty, self.stopy)]
        )

        if inverted_x:
            self.ax.invert_xaxis()
        if inverted_y:
            self.ax.invert_yaxis()
        self.redraw()  # force re-draw

    def move_draw(self, event):
        self.stopx = event.xdata
        self.stopy = event.ydata

        self.rect.set_width(self.stopx - self.startx)
        self.rect.set_height(self.stopy - self.starty)

        # self.ax.figure.canvas.draw() # force re-draw
        canvas = self.ax.figure.canvas
        canvas.restore_region(self.background)
        self.ax.draw_artist(self.rect)
        canvas.blit(self.ax.bbox)

    def start_move(self, event):
        self.moving = True
        (self.startx, self.starty) = self.to_log(event.xdata, event.ydata)

        self.connect_move()

    def stop_move(self, event):
        self.disconnect_move()

        if event.inaxes == self.ax:
            self.move_move(event)
        self.moving = False

    def move_move(self, event):
        (stopx, stopy) = self.to_log(event.xdata, event.ydata)

        dx = stopx - self.startx
        dy = stopy - self.starty

        (xlim, ylim) = self.to_log(self.ax.get_xlim(), self.ax.get_ylim())

        xlim -= dx
        ylim -= dy

        (xlim, ylim) = self.from_log(xlim, ylim)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        self.redraw()


class ToggleControl:
    """
    Add curve toggling feature to the legend:
    By clicking on the curves in the legend, the corresponding curves will turn
    on and off.
    """

    def __init__(self, ax):
        if hasattr(ax, "ll_toggle_ctrl"):
            ax.ll_toggle_ctrl.disconnect()
            del ax.ll_toggle_ctrl
        self.ax = ax

        def saveHandles(*args, **kwargs):
            handles, *_ = mlegend._parse_legend_args(
                [self.ax], *args, **kwargs
            )
            self.leg_handles = handles

        def doUpdate(ret, *args, **kwargs):
            self.updateDict()
            return ret

        self.ax.legend = wrap(
            self.ax.legend, before=saveHandles, after=doUpdate
        )

        self.leg = None
        self.leg_handles = None
        self.updateDict()
        self.connect()
        self.motion = False
        self.lastline = None

        ax.ll_toggle_ctrl = self

    @classmethod
    def remove(cls, ax):
        if hasattr(ax, "ll_toggle_ctrl"):
            ax.ll_toggle_ctrl.disconnect()
            ax.legend = unwrap(ax.legend)
            del ax.ll_toggle_ctrl

    def connect(self):
        "Attach the callbacks"
        connect = self.ax.figure.canvas.mpl_connect
        self.ax.set_picker(True)
        self.pick_cid = connect("pick_event", self.onpick_fun)

    def connect_motion(self):
        "Attach the drawing callbacks"
        connect = self.ax.figure.canvas.mpl_connect
        self.but_rel_cid = connect("button_release_event", self.but_rel_fun)
        self.motion_cid = connect("motion_notify_event", self.motion_fun)
        self.motion = True

    def disconnect_motion(self):
        "Remove the drawing callbacks"
        if self.motion:
            disconnect = self.ax.figure.canvas.mpl_disconnect
            disconnect(self.but_rel_cid)
            disconnect(self.motion_cid)
            self.motion = False

    def disconnect(self):
        "Remove the callbacks"
        disconnect = self.ax.figure.canvas.mpl_disconnect
        disconnect(self.pick_cid)
        if self.leg is not None:
            self.leg.remove_callback(self.leg_cid)
        self.disconnect_motion()

    def updateDict(self):
        # print("DEBUG: updating legend dict")
        self.lineDict = dict()

        leg = self.ax.get_legend()
        if leg != self.leg:
            if self.leg is not None:
                self.leg.remove_callback(self.leg_cid)
            if leg is not None:
                self.leg_cid = leg.add_callback(self.updateDict)
                self.leg = leg
        if leg is None:
            return
        if self.leg_handles is not None:
            lines = self.leg_handles
        else:
            lines = self.ax.get_lines()
        if hasattr(leg, "ll_mapping"):
            mapping = leg.ll_mapping
        else:
            mapping = [None for _ in lines]
        for legline, origline, mappings in zip(
            leg.get_lines(), lines, mapping
        ):
            legline.set_visible(True)
            if origline.get_visible():
                legline.set_alpha(1.0)
            else:
                legline.set_alpha(0.2)
            legline.set_picker(True)
            legline.set_pickradius(5)  # 5 pts tolerance
            if mappings is None:
                self.lineDict[legline] = [origline]
            else:
                self.lineDict[legline] = mappings
        self.ax.figure.canvas.draw()

    def toggle(self, legline):
        # If the legend has changed, rebuild the mapping dictionary
        if self.ax.get_legend() != self.leg:
            self.updateDict()
        origlines = self.lineDict.get(legline)
        if origlines is not None:
            vis = not origlines[0].get_visible()
            for origline in origlines:
                origline.set_visible(vis)
            # Change the alpha on the line in the legend so we can see what
            # lines have been toggled
            if vis:
                legline.set_alpha(1.0)
            else:
                legline.set_alpha(0.2)
            self.ax.figure.canvas.draw()
            return True
        return False

    def onpick_fun(self, event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility

        if event.mouseevent.button != MouseButton.LEFT:
            return
        legline = event.artist
        self.lastline = legline

        if self.toggle(legline):
            self.connect_motion()

    def but_rel_fun(self, event):
        if event.button != MouseButton.LEFT:
            return
        self.disconnect_motion()

    def motion_fun(self, event):
        if event.button != MouseButton.LEFT:
            self.disconnect_motion()
        for legline in self.leg.get_lines():
            if legline.contains(event)[0] and legline != self.lastline:
                self.lastline = legline
                self.toggle(legline)


class Marker:
    _marker = None

    def __init__(self, ax, formatter, name):
        if name == "D":
            self._marker = self.createDiffMarker(ax, formatter)
        else:
            self._marker = self.createDefaultMarker(ax, formatter)
        self._marker.set_visible(False)

    # Forward unkown attributes to the marker
    def __getattr__(self, name):
        # print(name)
        return getattr(self._marker, name)

    # Forward unkown attributes to the marker
    def __setattr__(self, name, value):
        # print('s: ', name)
        if name in self.__dir__():
            return object.__setattr__(self, name, value)
        else:
            return setattr(self._marker, name, value)

    @staticmethod
    def createDefaultMarker(ax, formatter):
        return ax.annotate(
            formatter,
            xy=(0, 0),
            ha="right",
            xytext=(-20, -20),
            textcoords="offset points",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="arc3,rad=0", color="black"
            ),
        )

    @staticmethod
    def createDiffMarker(ax, formatter):
        return ax.annotate(
            formatter,
            xy=(0, 0),
            ha="center",
            xytext=(0, 0),
            textcoords="offset points",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", fc="blue", alpha=0.5),
        )


class defaultDictKey(dict):
    def __init__(self, factory):
        self.factory = factory

    def __getitem__(self, key):
        if key not in self:
            self[key] = self.factory(key)
        return super().__getitem__(key)


class MarkerControl:
    """
    Add marker support.
    Press A/B to place markers, ctrl+e to remove them.
    Use the mouse to drag them around
    """

    def __init__(self, ax):
        if hasattr(ax, "ll_marker_ctrl"):
            ax.ll_marker_ctrl.disconnect()
            del ax.ll_marker_ctrl
        self.ax = ax

        self.markers = defaultDictKey(
            lambda name: Marker(ax, self.formatter, name)
        )

        self.connect()

        ax.ll_marker_ctrl = self

    @classmethod
    def remove(cls, ax):
        if hasattr(ax, "ll_marker_ctrl"):
            ax.ll_marker_ctrl.disconnect()
            del ax.ll_marker_ctrl

    @staticmethod
    def formatter(x, y, ax=None, delta=False, verbose=False, label=None):
        if ax is not None:
            xformat = ax.xaxis.get_major_formatter()
            yformat = ax.yaxis.get_major_formatter()

            if delta:
                if hasattr(xformat, "offset"):
                    x += xformat.offset
                if hasattr(yformat, "offset"):
                    y += yformat.offset
                if verbose:
                    dxy = x / y
                    dyx = y / x

                    if isinstance(xformat, ticker.EngFormatter):
                        xunit = xformat.unit
                    else:
                        xunit = 1
                    if isinstance(yformat, ticker.EngFormatter):
                        yunit = yformat.unit
                    else:
                        yunit = 1
                    if xunit == 1 and yunit == 1:
                        dxyformat = SCALAR_FORMATTER
                        dyxformat = SCALAR_FORMATTER
                    else:
                        dxyformat = EngFormatter(
                            unit=UNITS_DIV.get(
                                (xunit, yunit), f"[{xunit}/{yunit}]"
                            )
                        )
                        dyxformat = EngFormatter(
                            unit=UNITS_DIV.get(
                                (yunit, xunit), f"[{yunit}/{xunit}]"
                            )
                        )
                    dxy = dxyformat.format_data(dxy)
                    dyx = dyxformat.format_data(dyx)
            x = xformat.format_data_short(x)
            y = yformat.format_data_short(y)
        else:
            x = "{0.2g}".format(x)
            y = "{0.2g}".format(y)

        # Printing the format
        if delta:
            res = f"dx: {x}\ndy: {y}"
            if verbose:
                res += f"\ndx/dy: {dxy}\ndy/dx: {dyx}"
        else:
            res = f"x: {x}\ny: {y}"

        if label:
            res = f"{label}\n{res}"

        return res

    def connect(self):
        "Attach the callbacks"
        connect = self.ax.figure.canvas.mpl_connect
        self.key_cid = connect("key_press_event", self.key_fun)

    def disconnect(self):
        "Remove the callbacks"
        disconnect = self.ax.figure.canvas.mpl_disconnect
        disconnect(self.key_cid)
        for m in self.markers:
            self.markers[m].remove()
        self.markers.clear()

    def key_fun(self, event):
        if event.inaxes != self.ax:
            return
        if event.key == "a":
            self.place_marker(event, self.markers["A"])
        elif event.key == "b":
            self.place_marker(event, self.markers["B"])
        elif event.key == "A":
            self.place_marker(event, self.markers["A"], snap=False)
        elif event.key == "B":
            self.place_marker(event, self.markers["B"], snap=False)
        elif event.key == "d":
            self.place_delta_marker(
                event, self.markers["D"], self.markers["A"], self.markers["B"]
            )
        elif event.key == "D":
            self.place_delta_marker(
                event,
                self.markers["D"],
                self.markers["A"],
                self.markers["B"],
                verbose=True,
            )
        elif event.key == "ctrl+e":
            for m in self.markers:
                self.markers[m].remove()
            self.markers.clear()
            event.canvas.draw()

    def place_marker(self, event, marker, snap=True):
        if not snap:
            x, y = event.xdata, event.ydata
        else:
            x, y, label = self.snap(event.xdata, event.ydata)
        marker.xy = x, y
        marker.set_text(self.formatter(x, y, self.ax, label=label))
        marker.set_visible(True)
        event.canvas.draw()

    def place_delta_marker(self, event, marker, m1, m2, verbose=False):
        x1, y1 = m1.xy
        x2, y2 = m2.xy
        x = x2 - x1
        y = y2 - y1

        marker.xy = event.xdata, event.ydata
        marker.set_text(self.formatter(x, y, self.ax, True, verbose))
        marker.set_backgroundcolor("blue")
        marker.xyann = (0, 0)
        marker.set_visible(True)
        event.canvas.draw()

    def snap(self, x, y):
        try:
            # Flatten all the data and save start indicies
            data = []
            label_lst = []
            len_lst = []
            start_index = 0
            for line in self.ax.lines:
                if line.get_visible():
                    data_tba = line.get_data()
                    start_index += np.shape(data_tba)[1]
                    len_lst.append(start_index)
                    data.append(data_tba)
                    label = line.get_label()
                    if label.startswith("_"):
                        label = None
                    label_lst.append(label)

            data = np.hstack(data).T

            trans = self.ax.lines[0].get_transform()
            sdata = trans.transform(data)
            x, y = trans.transform([x, y])
            idx = np.nanargmin(((sdata - (x, y)) ** 2).sum(axis=-1))
            len_lst = [0] + len_lst
            label = ""
            # Search where the label should be in the start indices.
            for index, _ in enumerate(len_lst[:-1]):
                if len_lst[index] <= idx < len_lst[index + 1]:
                    label = label_lst[index]

            xdata = data[idx, 0]
            ydata = data[idx, 1]

        except TypeError:
            xdata, ydata = x, y
        return xdata, ydata, label
