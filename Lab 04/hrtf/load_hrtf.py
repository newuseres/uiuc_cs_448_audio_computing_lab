import numpy as np
from pathlib import Path
import re

_AZ_RE = re.compile(r"e(?P<az>\d{3})a\.dat$", re.IGNORECASE)

def _nearest(value, candidates):
    """Return the candidate closest to value."""
    candidates = np.array(sorted(candidates), dtype=float)
    idx = int(np.argmin(np.abs(candidates - value)))
    return float(candidates[idx])

def load_hrtf(ad, ed, base_dir=None, verbose=False):
    """
    Load interleaved (L,R,L,R,...) HRTF impulse responses from .dat files.
    
    Parameters
    ----------
    ad : float
        Azimuth in degrees. 0 = front. Negative = left (mirrored convention used below).
    ed : float
        Elevation in degrees.
    base_dir : str or Path or None
        Path to the 'compact' folder. If None, resolves relative to this file:
        <this_file>/compact
    verbose : bool
        Print chosen elevation and azimuth and file path.

    Returns
    -------
    left, right : np.ndarray
        1-D float arrays (typically length 128).
    """
    # Resolve base path robustly
    if base_dir is None:
        # assumes this file lives in .../hrtf/load_hrtf.py and compact is .../hrtf/compact
        base = Path(__file__).resolve().parent / "compact"
    else:
        base = Path(base_dir).expanduser().resolve()

    # Normalize azimuth to [-180, 180]
    ad = float(ad) % 360.0
    if ad > 180.0:
        ad -= 360.0

    # Mirror negative azimuths (keep your original convention)
    fl = ad < 0
    a_target = abs(ad) if fl else ad  # target in [0, 180]

    # Pick elevation folder: nearest available elev*
    elev_dirs = sorted([d for d in base.glob("elev*") if d.is_dir()])
    if not elev_dirs:
        raise FileNotFoundError(f"No elevation folders found under: {base}")

    elev_vals = []
    for d in elev_dirs:
        try:
            elev_vals.append(int(d.name.replace("elev", "")))
        except ValueError:
            pass

    if not elev_vals:
        raise FileNotFoundError(f"Found elev dirs but couldn't parse elevation numbers under: {base}")

    e_chosen = int(_nearest(ed, elev_vals))
    elev_path = base / f"elev{e_chosen}"

    # List dat files in that elevation folder
    dat_files = sorted(elev_path.glob(f"H{e_chosen}e* a.dat".replace(" ", "")))  # tolerant-ish
    if not dat_files:
        # fallback: just take all .dat in folder
        dat_files = sorted(elev_path.glob("*.dat"))

    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in: {elev_path}")

    # Parse available azimuths from filenames
    az_to_file = {}
    for f in dat_files:
        m = _AZ_RE.search(f.name)
        if not m:
            continue
        az = int(m.group("az"))
        az_to_file[az] = f

    if not az_to_file:
        raise FileNotFoundError(
            f"Could not parse azimuth from filenames in {elev_path}. "
            f"Expected pattern like ...e###a.dat"
        )

    available_az = sorted(az_to_file.keys())
    a_chosen = int(_nearest(a_target, available_az))
    fpath = az_to_file[a_chosen]

    if verbose:
        print(f"Requested (ad, ed)=({ad:.1f}, {ed:.1f}) -> using elev {e_chosen}, az {a_chosen}, file {fpath}")

    # Load interleaved samples and split
    h = np.fromfile(fpath, dtype=">i2").astype(np.float64) / 32768.0

    # Interleaving convention:
    # original code returned different channel order depending on mirroring.
    # Keep same behavior:
    if fl:
        # mirrored: swap L/R
        left = h[::2]
        right = h[1::2]
    else:
        left = h[1::2]
        right = h[::2]

    return left, right