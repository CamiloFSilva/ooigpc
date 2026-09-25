"""
signal/envelope.py — Amplitude envelope extraction from a band-pass signal.

Given a narrow-band pressure signal p(t) centred near ω₀, the amplitude
envelope A(t) is extracted as the magnitude of the analytic signal:

    p_a(t) = p(t) + i·H{p(t)}
    A(t) = |p_a(t)|

where H{·} denotes the Hilbert transform.

For experimental data the signal should first be band-pass filtered
(see :mod:`ooigpc.signal.bandpass`) to isolate the thermoacoustic mode.

References
----------
Silva, C.F. et al. (2026). *Output-only identification of thermoacoustic
systems using polynomial chaos expansions.* Combust. Flame.
Supplementary material §1 (data pre-processing).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

__all__ = [
    "hilbert_envelope",
    "running_rms_envelope",
]


def hilbert_envelope(signal: np.ndarray) -> np.ndarray:
    """Extract the amplitude envelope via the Hilbert transform.

    Parameters
    ----------
    signal:
        1-D pressure (or displacement) time series, already band-pass
        filtered to isolate a single thermoacoustic mode.

    Returns
    -------
    numpy.ndarray
        Amplitude envelope A(t), same shape as *signal*.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 10000)
    >>> p = np.sin(2 * np.pi * 120 * t)   # pure 120 Hz tone
    >>> A = hilbert_envelope(p)
    >>> np.allclose(A, 1.0, atol=1e-3)    # envelope ≈ 1 everywhere
    True
    """
    analytic = hilbert(np.asarray(signal, dtype=float))
    return np.abs(analytic)


def running_rms_envelope(
    signal: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute the running-RMS amplitude envelope.

    An alternative to the Hilbert envelope for signals with strong
    harmonics or broad-band noise.  Each output sample is the RMS of
    the input over a sliding window of length *window*.

    Parameters
    ----------
    signal:
        1-D signal array.
    window:
        Window length (samples).  Should span roughly one oscillation
        period, e.g. ``window = int(fs / f0)``.

    Returns
    -------
    numpy.ndarray
        Running-RMS envelope, same length as *signal*.
    """
    sig = np.asarray(signal, dtype=float)
    sq = sig**2
    # Use cumsum for an O(N) sliding window mean
    cs = np.cumsum(np.concatenate([[0.0], sq]))
    half = window // 2
    # Pad boundaries with edge values
    rms = np.sqrt(np.maximum(
        (cs[window:] - cs[:-window]) / window, 0.0
    ))
    # Align to centre of window and pad edges
    pad_left = np.full(half, rms[0])
    pad_right = np.full(len(sig) - len(rms) - half, rms[-1])
    return np.concatenate([pad_left, rms, pad_right])
