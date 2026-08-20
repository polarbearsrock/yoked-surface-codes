import collections
import math
from typing import Mapping, Tuple, Iterable, Sequence, List

import numpy as np
import sinter


def min_sample_extrapolate(
        hits: Mapping[float, int],
        n: float,
):
    """Extrapolates a histogram to the histogram of the min of n iid draws.

    Treats `hits` as an empirical distribution and returns the expected
    histogram (same bin locations, same total count) of min(X_1, ..., X_n)
    for n independent draws from it, via S_min(x) = S(x)**n on the survival
    function. Used e.g. to predict the worst gap among n patches from
    single-patch gap data. n may be fractional.
    """
    n_hits = sum(hits.values())
    xs = np.array(sorted({k: v for k, v in hits.items()}))
    hs = np.array([hits[x] for x in xs])
    ps = hs / n_hits

    cs = np.cumsum(ps)
    cs = np.maximum(0, 1 - cs)
    cs **= n
    cs = np.maximum(0, 1 - cs)
    ps = np.diff(cs, prepend=0)

    result = {}
    for k in range(len(xs)):
        result[xs[k]] = ps[k] * n_hits
    return result


def histogram_cumulative_meet_in_the_middle(
        *,
        total_hits: int,
        hits: Mapping[float, int],
) -> Tuple[Iterable[float], Iterable[float]]:
    """Returns the two-sided cumulative distribution of a histogram.

    At each support point x the returned value is the smaller of P(X <= x)
    and P(X >= x) (both relative to total_hits), so both tails of the
    distribution show up when plotted on a log axis.
    """
    hits = {k: v for k, v in hits.items() if v}
    xs = np.array(sorted(hits.keys()))
    hs = np.array([hits[x] for x in xs])
    ps = hs / total_hits
    return xs, np.minimum(np.cumsum(ps), np.cumsum(ps[::-1])[::-1])


def histogram_bucket_holds(
        *,
        hits: Mapping[float, int],
        bucket_width: float,
) -> Tuple[Sequence[float], Sequence[float]]:
    """Turns a histogram into a sliding-window ('bucket hold') count curve.

    For each support point x, emits the number of hits inside a window of
    size bucket_width starting at x, twice: once including a hit exactly at
    x and once just after it, producing the vertical steps where the window
    boundary crosses a data point. Zero-count support points are added one
    bucket_width before each hit so the curve drops back between clusters,
    and the returned x coordinates are shifted to the window centers.
    """
    hits = {k: v for k, v in hits.items() if v}
    for k in list(hits.keys()):
        hits.setdefault(k - bucket_width, 0)
    dat = sorted(hits.items())
    in_xs = [e[0] for e in dat]
    in_hits = [e[1] for e in dat]

    out_xs: List[float] = []
    out_ys: List[float] = []
    for k in range(len(in_xs)):
        x = in_xs[k]
        k2 = k
        while k2 < len(in_xs) and in_xs[k2] - (x - 1e-8) < bucket_width:
            k2 += 1
        out_xs.append(x)
        out_ys.append(float(np.sum(in_hits[k:k2])))

        k2 = k + 1
        while k2 < len(in_xs) and in_xs[k2] - (x + 1e-8) < bucket_width:
            k2 += 1
        out_xs.append(x)
        out_ys.append(float(np.sum(in_hits[k + 1:k2])))

    out_xs = [e + bucket_width / 2 for e in out_xs]

    return out_xs, out_ys


def histogram_cosine_convolve(
        *,
        hits: Mapping[int, int],
        bucket_width: int,
) -> Tuple[Sequence[float], Sequence[float]]:
    """Smooths an integer-keyed histogram with a raised-cosine kernel.

    The support is resampled at 1/8 steps (mass stays at the integer
    positions) and convolved with a cos+1 kernel roughly 1.25*bucket_width
    wide, then trimmed back to the resampled support.
    """
    assert all(k == int(k) for k in hits.keys())
    min_hit = int(min(hits.keys(), default=0))
    max_hit = int(max(hits.keys(), default=-1))
    xs = []
    for k in range(min_hit - bucket_width, max_hit + bucket_width):
        for d in range(8):
            xs.append(k + d/8)
    xs = np.array(xs, dtype=np.float64)
    ys = np.array([hits.get(x, 0) for x in xs], dtype=np.float64)
    convolve = np.cos(np.linspace(-np.pi, np.pi, num=math.ceil(bucket_width)*10 + 1)) + 1
    ys = np.convolve(ys, convolve)
    assert not any(ys[:bucket_width*5])
    assert not any(ys[-bucket_width*5:])
    ys = ys[bucket_width*5:-bucket_width*5]
    assert len(xs) == len(ys)
    return xs, ys


def curve_to_curve_between_midpoints(
        *,
        xs: Sequence[float],
        ys: Sequence[float],
) -> Tuple[Sequence[float], Sequence[float]]:
    """Resamples a curve at the midpoints between its points (ends kept)."""
    if len(xs) == 0:
        return xs, ys
    out_xs = []
    out_ys = []
    out_xs.append(xs[0])
    out_ys.append(ys[0])
    for k in range(1, len(xs)):
        out_xs.append((xs[k - 1] + xs[k]) / 2)
        out_ys.append((ys[k - 1] + ys[k]) / 2)
    out_xs.append(xs[-1])
    out_ys.append(ys[-1])
    return out_xs, out_ys


def curve_rescaled_to_target_area(
        *,
        expected_total: float,
        xs: Sequence[float],
        ys: Sequence[float],
) -> Tuple[Sequence[float], Sequence[float]]:
    """Rescales a curve so its trapezoidal area equals expected_total."""
    return xs, np.array(ys) * expected_total / area_under_curve(xs, ys)


def histogram_slope_holds(
        *,
        total_hits: int,
        hits: Mapping[float, int],
) -> Tuple[Iterable[float], Iterable[float]]:
    """Estimates a density curve from a histogram via CDF slopes.

    The slope of the empirical CDF between each pair of adjacent support
    points is emitted as a horizontal 'hold' segment spanning that pair, and
    the result is rescaled so its area equals the histogram's total
    probability mass (relative to total_hits).
    """
    hits = {k: v for k, v in hits.items() if v}
    xs = np.array(sorted(hits.keys()))
    hs = np.array([hits[x] for x in xs])
    ps = hs / total_hits
    cs = np.cumsum(ps)

    expected_total_p = sum(hits.values()) / total_hits
    transition_xs = []
    transition_ys = []
    for k in range(len(xs) - 1):
        x0 = xs[k]
        x1 = xs[k + 1]
        y0 = cs[k]
        y1 = cs[k + 1]
        dx = x1 - x0
        dy = y1 - y0
        slope = dy / dx

        transition_xs.append(x0)
        transition_ys.append(slope)
        transition_xs.append(x1)
        transition_ys.append(slope)

    transition_ys = np.array(transition_ys) * expected_total_p / area_under_curve(transition_xs, transition_ys)

    return transition_xs, transition_ys


def area_under_curve(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Trapezoid-rule area; returns 1e-10 instead of 0 so callers can divide."""
    total = 0
    for k in range(len(xs) - 1):
        x0 = xs[k]
        x1 = xs[k + 1]
        y0 = ys[k]
        y1 = ys[k + 1]
        dx = x1 - x0
        total += dx * (y0 + y1) / 2
    if total == 0:
        total = 1e-10
    return total


def histogram_slope_centers(
        *,
        total_hits: int,
        hits: Mapping[float, int],
) -> Tuple[Iterable[float], Iterable[float]]:
    """Like `histogram_slope_holds`, but smoother: the CDF slope is taken
    across a 5-point span and plotted at the span's center.
    """
    hits = {k: v for k, v in hits.items()}
    xs = np.array(sorted(hits.keys()))
    hs = np.array([hits[x] for x in xs])
    ps = hs / total_hits
    cs = np.cumsum(ps)

    expected_total_p = sum(hits.values()) / total_hits
    center_xs = []
    center_ys = []
    for k in range(len(xs) - 5):
        x0 = xs[k]
        x1 = xs[k + 5]
        y0 = cs[k]
        y1 = cs[k + 5]
        dx = x1 - x0
        dy = y1 - y0
        slope = dy / dx

        center_xs.append((x0 + x1) / 2)
        center_ys.append(slope)

    center_ys = np.array(center_ys) * expected_total_p / area_under_curve(center_xs, center_ys)

    return center_xs, center_ys


def with_unsigned_gap(
        stat: sinter.TaskStats,
        *,
        invert_success_if_negative: bool,
        write_success_into_sign: bool,
) -> sinter.TaskStats:
    """Re-encodes a stat's signed-gap custom_counts histogram.

    The input keys look like 'C<gap>' or 'E<gap>': shots decoded correctly or
    incorrectly, binned by complementary gap (see `yoked.gap`). A negative
    gap means the decoder preferred the decode with the comparison detector
    flipped.

    Args:
        stat: The stats whose custom_counts should be re-encoded.
        invert_success_if_negative: Applies the comparison (yoke) check to
            the success classification: a 'C' shot with a negative gap
            becomes an error, because acting on the comparison decode's
            preferred value would flip an otherwise-correct result. An
            'E-#' shot stays an error. The gap magnitude is kept either way.
        write_success_into_sign: Re-signs the gap magnitude to encode success
            instead: errors get a negative gap and are then relabeled 'C',
            so every shot lands in a single 'C<signed gap>' key namespace.
            A zero-gap error is stored as -0.01, a sentinel just below zero,
            so the failure survives when the sign is the only bit carrying it.

    Zero-gap successes are split as evenly as integer counts permit between
    'C0.0' and 'E0.0', since a zero gap means the two decodes tied and the
    decoder effectively guessed. Any odd remainder is assigned to success.
    """
    new_custom_counts = collections.Counter()
    for k, v in stat.custom_counts.items():
        success = k[0] == 'C'
        g = float(k[1:])
        if invert_success_if_negative and g < 0 and success:  # 'E-#' is still an error.
            success = not success
        g = abs(g)
        if write_success_into_sign and not success:
            g *= -1
            if g == 0:
                g = -0.01
            success = True
        if g == 0 and success:  # Split 'C0' evenly into successes and failures.
            successes = (v + 1) // 2
            new_custom_counts[f'C{g}'] += successes
            new_custom_counts[f'E{g}'] += v - successes
            continue
        new_custom_counts[f'{"EC"[success]}{g}'] += v
    return sinter.TaskStats(
        strong_id=stat.strong_id,
        decoder=stat.decoder,
        json_metadata=stat.json_metadata,
        shots=stat.shots,
        errors=stat.errors,
        discards=stat.discards,
        seconds=stat.seconds,
        custom_counts=new_custom_counts,
    )
