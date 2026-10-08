"""Consistent, unlabeled minor ticks for manuscript logarithmic axes."""
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter


def add_log_minor_ticks(fig):
    """Keep linear axes, major ticks, limits, and plotted quantities intact."""
    for ax in fig.findobj(Axes):
        if not ax.axison:
            continue
        for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            if axis.get_scale() != "log":
                continue
            limits = getattr(ax, f"get_{name}lim")()
            decades = abs(np.log10(limits[1] / limits[0]))
            # Include unlabeled decade marks where major labels skip decades.
            # Sparse subdivisions keep very wide dynamic ranges legible.
            subs = (1, 2, 5) if decades > 6 else tuple(range(1, 10))
            if not isinstance(axis.get_minor_locator(), FixedLocator):
                axis.set_minor_locator(LogLocator(base=10, subs=subs, numticks=100))
            axis.set_minor_formatter(NullFormatter())
            # Match the major ticks' visible sides (including twin axes).
            major = axis.get_major_ticks()[0]
            sides = dict(bottom=major.tick1line.get_visible(), top=major.tick2line.get_visible()) if name == "x" else dict(left=major.tick1line.get_visible(), right=major.tick2line.get_visible())
            ax.tick_params(axis=name, which="minor", direction="in", length=2,
                           width=.5, **sides)
