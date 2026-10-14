"""
Classed (discrete) colour scales for every map, animation and figure.

Why classes: the eye does not read a continuous colour ramp as a number; it
sees a handful of distinguishable steps.  So every magnitude is binned into a
few classes with round, labelled breaks and one colour per class.

Rules (from the project's data-viz method):

* Magnitude (sequential): ONE hue, light -> dark (blue ramp).
* Change / anomaly / trend (diverging): blue (less) <-> red (more) with a
  neutral grey class around zero; the same number of steps on each arm.
* Increase / decrease / not significant (categorical): red / blue / grey.

The blue ramp is the reference palette's sequential ramp; the red ramp was
derived from it in OKLCH with the same lightness at every step (hue of the
reference categorical red, maximum in-gamut chroma), so both arms of a
diverging scale weigh the same.
"""

from __future__ import annotations

import math

import numpy as np

BLUE = {
    100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef", 300: "#6da7ec",
    350: "#5598e7", 400: "#3987e5", 450: "#2a78d6", 500: "#256abf", 550: "#1c5cab",
    600: "#184f95", 650: "#104281", 700: "#0d366b",
}
RED = {  # same OKLCH lightness as BLUE, hue 25 deg
    100: "#fed4d0", 150: "#f9c0bb", 200: "#f7aba4", 250: "#f1968e", 300: "#ed7f78",
    350: "#e66963", 400: "#e14c4a", 450: "#d03b3c", 500: "#ba3334", 550: "#a6272a",
    600: "#902123", 650: "#7d171a", 700: "#671215",
}
NEUTRAL = "#e4e3de"  # diverging midpoint, visible on a white map background
NOT_SIGNIFICANT = "#d9d8d2"  # recedes behind the coloured classes
INCREASE = RED[450]
DECREASE = BLUE[450]

_NICE = (1.0, 2.0, 2.5, 5.0)


def nice_step(raw: float) -> float:
    """Smallest 'round' number (1, 2, 2.5, 5 x 10^k) >= raw."""
    if raw <= 0 or not np.isfinite(raw):
        return 1.0
    k = math.floor(math.log10(raw))
    for m in _NICE + (10.0,):
        if m * 10**k >= raw - 1e-12:
            return m * 10**k
    return 10.0 ** (k + 1)


def nice_round(x: float) -> float:
    """Round to 2 significant digits on the 1/2/2.5/5 grid of its decade."""
    if x <= 0:
        return 0.0
    k = math.floor(math.log10(x))
    base = 10 ** (k - 1)
    return round(x / (5 * base)) * 5 * base or base


def _ramp(ramp: dict[int, str], n: int, first: int = 150, last: int = 700) -> list[str]:
    keys = sorted(k for k in ramp if first <= k <= last)
    idx = np.round(np.linspace(0, len(keys) - 1, n)).astype(int)
    return [ramp[keys[i]] for i in idx]


def _fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        out = f"{int(round(v))}"
    else:
        out = f"{v:g}"
    return out.replace("-", "−")


class Classes:
    """Boundaries + colours + labels; builds the matplotlib cmap/norm."""

    def __init__(self, boundaries, colors, labels, extend="neither"):
        self.boundaries = list(boundaries)
        self.colors = list(colors)
        self.labels = list(labels)
        self.extend = extend

    def cmap_norm(self):
        from matplotlib.colors import BoundaryNorm, ListedColormap

        cols = list(self.colors)
        under = over = None
        if self.extend in ("min", "both"):
            under = cols.pop(0)
        if self.extend in ("max", "both"):
            over = cols.pop(-1)
        cmap = ListedColormap(cols)
        if under:
            cmap.set_under(under)
        if over:
            cmap.set_over(over)
        cmap.set_bad(alpha=0)
        return cmap, BoundaryNorm(self.boundaries, cmap.N)

    def colorbar(self, fig, mappable, cax, label: str = ""):
        """Discrete colourbar: equal-width boxes, ticks on the class breaks."""
        n_boxes = len(self.boundaries) + 1
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal",
                          extend=self.extend, spacing="uniform",
                          ticks=self.boundaries, extendfrac=1 / max(n_boxes - 2, 1),
                          extendrect=True)
        if getattr(self, "zero_class", False):
            # the first break only separates the "0" class: label the box, not the edge
            ticks = self.boundaries[1:]
            cb.set_ticks(ticks)
            cb.ax.set_xticklabels([_fmt(b) for b in ticks])
            cb.ax.text(-0.5 / max(n_boxes - 2, 1), -0.35, "0", transform=cb.ax.transAxes,
                       ha="center", va="top", fontsize=9)
        else:
            cb.ax.set_xticklabels([_fmt(b) for b in self.boundaries])
        cb.ax.tick_params(length=0, labelsize=9)
        cb.outline.set_visible(False)
        if label:
            cb.set_label(label, fontsize=10)
        return cb


def sequential_classes(
    values,
    *,
    n_classes: int = 6,
    method: str = "auto",
    percentiles: tuple[float, float] = (1.0, 99.0),
    ramp: dict[int, str] = BLUE,
    integer: bool | None = None,
    zero_class_share: float = 0.10,
) -> Classes:
    """
    One-hue classes for a magnitude, about ``n_classes`` of them.

    * ``"linear"``: equal round steps between the 1st and 99th percentiles;
      the first class is open below ("< b0") and the last open above.
    * ``"quantile"``: breaks at quantiles, rounded to round numbers, for
      skewed variables where equal steps would leave most of the map in one
      class.  ``"auto"`` picks it when the median is below 25 % of the
      99th percentile.
    * When at least ``zero_class_share`` of the values are exactly 0 they get
      their own class ("0"), and the other classes split the positive values.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        v = np.array([0.0, 1.0])
    if integer is None:
        integer = bool(np.allclose(v, np.round(v)))
    zero_share = float(np.mean(v == 0))
    zero_class = zero_share >= zero_class_share
    pos = v[v > 0] if zero_class else v
    if pos.size == 0:
        pos = np.array([1.0])
    lo, hi = (float(np.percentile(pos, q)) for q in percentiles)
    n = n_classes - (1 if zero_class else 0)
    if method == "auto":
        method = "quantile" if hi > 0 and float(np.median(pos)) < 0.25 * hi else "linear"

    if method == "linear":
        raw = max(hi - lo, 1e-9) / n
        if integer and raw <= 10:
            # small integer counts (e.g. applications): any whole step is readable,
            # and a coarse "round" step would lump most of the map into one class
            step = float(max(1, math.ceil(raw)))
        else:
            step = nice_step(raw)
        k0 = math.floor(lo / step) + 1
        inner = [step * k for k in range(k0, k0 + n - 1)]
    elif method == "quantile":
        qs = np.percentile(pos, np.linspace(percentiles[0], percentiles[1], n + 1)[1:-1])
        inner = sorted({nice_round(q) for q in qs if q > 0})
        if integer:
            inner = sorted({max(1.0, float(round(b))) for b in inner})
    else:
        raise ValueError("method must be 'auto', 'linear' or 'quantile'")

    inner = [b for b in inner if b > 0]
    if zero_class:
        first = 0.5 if integer else 1e-9
        bounds = [first] + [b for b in inner if b > first]
        labels = ["0"]
    else:
        bounds = inner
        labels = [f"< {_fmt(bounds[0])}"]
    pairs = list(zip(bounds[:-1], bounds[1:]))
    for i, (a, b) in enumerate(pairs):
        if zero_class and i == 0:
            a_txt = "1" if integer else ">0"
        else:
            a_txt = _fmt(a)
        labels.append(f"{a_txt}–{_fmt(b)}")
    labels += [f"≥ {_fmt(bounds[-1])}"]
    colors = _ramp(ramp, len(bounds) + 1)
    c = Classes(bounds, colors, labels, extend="both")
    c.zero_class = zero_class
    return c


def diverging_classes(
    values=None,
    *,
    absmax: float | None = None,
    n_per_arm: int = 4,
    percentile: float = 98.0,
) -> Classes:
    """
    Blue (negative) <-> grey (about zero) <-> red (positive), equal arms.

    Breaks: +-s/2 (neutral class), +-s, +-2s, ... with a round step ``s``;
    both outer classes are open-ended.
    """
    if absmax is None:
        a = np.abs(np.asarray(values, dtype=float).ravel())
        a = a[np.isfinite(a)]
        absmax = float(np.percentile(a, percentile)) if a.size else 1.0
    s = nice_step(absmax / (n_per_arm - 0.5))
    pos = [s / 2] + [s * k for k in range(1, n_per_arm)]
    bounds = [-b for b in reversed(pos)] + pos
    blues = _ramp(BLUE, n_per_arm, 250, 700)[::-1]  # darkest = most negative
    reds = _ramp(RED, n_per_arm, 250, 700)
    colors = blues + [NEUTRAL] + reds
    labels = (
        [f"< {_fmt(bounds[0])}"]
        + [f"{_fmt(a)} a {_fmt(b)}" for a, b in zip(bounds[:-1], bounds[1:])]
        + [f"> {_fmt(bounds[-1])}"]
    )
    return Classes(bounds, colors, labels, extend="both")
