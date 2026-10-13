import numpy as np

from viz_palette import BLUE, RED, diverging_classes, nice_step, sequential_classes


def _lightness(hexcolor):
    r, g, b = (int(hexcolor[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_ramps_monotone_and_matched():
    keys = sorted(BLUE)
    lb = [_lightness(BLUE[k]) for k in keys]
    lr = [_lightness(RED[k]) for k in keys]
    assert all(a > b for a, b in zip(lb, lb[1:]))
    assert all(a > b for a, b in zip(lr, lr[1:]))


def test_nice_step():
    assert nice_step(0.7) == 1.0
    assert nice_step(1.6) == 2.0
    assert nice_step(2.2) == 2.5
    assert nice_step(37) == 50.0


def test_sequential_linear_integer():
    v = np.random.default_rng(0).integers(4, 13, 10_000)
    c = sequential_classes(v, n_classes=6)
    assert c.extend == "both" and len(c.colors) == len(c.boundaries) + 1
    assert all(float(b).is_integer() for b in c.boundaries)
    cmap, norm = c.cmap_norm()
    assert cmap.N == len(c.boundaries) - 1


def test_sequential_zero_class():
    v = np.r_[np.zeros(6000), np.random.default_rng(1).integers(1, 500, 4000)]
    c = sequential_classes(v)
    assert c.zero_class and c.labels[0] == "0" and c.boundaries[0] == 0.5
    assert c.labels[1].startswith("1–")


def test_diverging_symmetric():
    c = diverging_classes(absmax=3.2)
    b = np.array(c.boundaries)
    assert np.allclose(b, -b[::-1])
    assert len(c.colors) == len(b) + 1
    mid = len(c.colors) // 2
    assert c.colors[mid] not in BLUE.values() and c.colors[mid] not in RED.values()
