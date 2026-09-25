"""
signal/bandpass.py — Band-pass filtering to isolate a thermoacoustic mode.

Before extracting the amplitude envelope (see :mod:`ooigpc.signal.envelope`),
the raw pressure signal should be filtered to retain only the frequency band
around the dominant thermoacoustic mode at f₀.

Two implementations are provided:

* :func:`bandpass_butter` — IIR Butterworth filter (low latency, suitable
  for long signals).
* :func:`bandpass_rect`   — Rectangular (brick-wall) spectral filter in the
  frequency domain, as used in the MATLAB reference script ``BP_filter_rect.m``.

References
----------
Silva, C.F. et al. (2026). *Output-only identification of thermoacoustic
systems using polynomial chaos expansions.* Combust. Flame.
Supplementary material §1.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

__all__ = [
    "bandpass_butter",
    "bandpass_rect",
]


def bandpass_butter(
    signal: np.ndarray,
    fs: float,
    f_low: float,
    f_high: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter.

    Uses ``sosfiltfilt`` for forward–backward filtering (linear phase,
    no group delay).

    Parameters
    ----------
    signal:
        1-D input signal.
    fs:
        Sampling frequency (Hz).
    f_low, f_high:
        Lower and upper cutoff frequencies (Hz).
    order:
        Butterworth filter order (applied twice by ``filtfilt``).

    Returns
    -------
    numpy.ndarray
        Filtered signal, same shape as *signal*.

    Examples
    --------
    >>> import numpy as np
    >>> fs, f0 = 10_000.0, 120.0
    >>> t = np.linspace(0, 1, int(fs))
    >>> p = np.sin(2*np.pi*f0*t) + 0.5*np.sin(2*np.pi*500*t)
    >>> p_filt = bandpass_butter(p, fs, f_low=80.0, f_high=160.0)
    """
    nyq = fs / 2.0
    sos = butter(order, [f_low / nyq, f_high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, np.asarray(signal, dtype=float))


def bandpass_rect(
    signal: np.ndarray,
    fs: float,
    f_low: float,
    f_high: float,
) -> np.ndarray:
    """Rectangular (brick-wall) spectral band-pass filter.

    Computes the FFT, zeroes out all frequency bins outside
    [*f_low*, *f_high*], and returns the real part of the IFFT.
    This matches the approach of ``BP_filter_rect.m``.

    Parameters
    ----------
    signal:
        1-D input signal.
    fs:
        Sampling frequency (Hz).
    f_low, f_high:
        Pass-band edges (Hz).

    Returns
    -------
    numpy.ndarray
        Filtered signal, same shape as *signal*.

    Notes
    -----
    The rectangular filter has Gibbs-ringing artefacts near sharp
    transitions; use :func:`bandpass_butter` for signals with transients.
    """
    sig = np.asarray(signal, dtype=float)
    n = len(sig)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.fft.rfft(sig)
    mask = (freq >= f_low) & (freq <= f_high)
    spec_filt = spec * mask
    filtered = np.fft.irfft(spec_filt, n=n)
    return filtered
