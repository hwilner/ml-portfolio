"""NYSIIS phonetic encoding, with a fallback for the un-installable `fuzzy` package.

The original notebook called `fuzzy.nysiis()`. The `fuzzy` distribution on PyPI
ships as a C extension and needs a working compiler plus Cython at install
time, so on many machines `pip install fuzzy` simply fails. That made the
notebook unrunnable for anyone who had not already installed it.

This module resolves the function once, preferring the original and falling
back to `abydos`, which is pure Python and installs everywhere:

    >>> from nysiis_compat import nysiis
    >>> nysiis("Pfister")
    'FASTAR'

**A warning about the fallback.** NYSIIS has several published
implementations and they do not agree on every name. `abydos` agrees with the
reference on cases like Macintosh/McIntosh and Pfister/Fister, but differs on
others (it gives Smith -> SNAT where some references give SMAD). If
`fuzzy` is available it is used and this concern disappears. Which is
*available* cannot change the notebook's conclusions either way - see the
caveat at the end of the notebook, which is the real limitation of this
approach.
"""

from __future__ import annotations

__all__ = ["nysiis", "BACKEND"]

try:  # the original, preferred
    from fuzzy import nysiis

    BACKEND = "fuzzy"
except ImportError:  # pragma: no cover - depends on the environment
    try:
        from abydos.phonetic import NYSIIS

        _encoder = NYSIIS()

        def nysiis(name: str) -> str:
            """Compute the NYSIIS phonetic key for a name.

            Args:
                name: Any string. Non-alphabetic characters are discarded.

            Returns:
                str: The uppercase phonetic key.
            """
            return _encoder.encode(name)

        BACKEND = "abydos"
    except ImportError:  # pragma: no cover
        nysiis = None
        BACKEND = None
