"""
In-process spherical forward model backed by the MOPSMAP single-particle
efficiency table (mopsmap_sphere_table/mopsmap_spheres_v1.nc; see the README
there for provenance and license). Drop-in replacement for
mopsmap_wrapper.Model limited to integrated ext/sca outputs for spherical
particles: same call signature, same input conventions (dndlogdp in m^-3,
dpg diameters in micrometer, coefficients returned in m^-1), same PSD
representation (piecewise-linear dN/dlogDp between bin centers, zero
outside), ~100x faster (no subprocess, no temp files).

Validated against direct high-resolution Mie to <=0.21% (worst case; PSDs
from GSD 1.05 to 2.0, off-node CRIs, 450-700 nm) and against the MOPSMAP
executable itself (which agrees with exact Mie to ~0.2%).

Returns only the keys ISARA uses: 'ext_coeff_{w}_m-1' and 'ssa_{w}'.
"""
import os

import numpy as np
import netCDF4 as nc

_TABLE = None
_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "mopsmap_sphere_table", "mopsmap_spheres_v1.nc")
N_FINE_PER_BIN = 40   # fine quadrature points per PSD bin interval
N_FINE_MIN = 1000


def load_table(path=None):
  """Load (and cache) the efficiency table; returns the module-level dict."""
  global _TABLE
  if _TABLE is None or (path is not None and _TABLE["path"] != path):
    p = path or _TABLE_PATH
    with nc.Dataset(p) as d:
      _TABLE = {
        "path": p,
        "mreal": d["mreal"][:].filled(np.nan),
        "mimag": d["mimag"][:].filled(np.nan),
        "logsp": np.log(d["sizepara"][:].filled(np.nan)),
        "qext": d["qext"][:].filled(np.nan).astype(np.float64),
        "qsca": d["qsca"][:].filled(np.nan).astype(np.float64),
      }
      _TABLE["logmi"] = np.log(np.maximum(_TABLE["mimag"], 1e-9))
  return _TABLE


def _corner_weights(grid, v):
  i = int(np.clip(np.searchsorted(grid, v) - 1, 0, len(grid) - 2))
  f = (v - grid[i]) / (grid[i + 1] - grid[i])
  return i, f


def _q_at(t, rri, iri, logx):
  """qext, qsca at log size parameters logx for one CRI (bilinear mr/log-mi)."""
  i, fr = _corner_weights(t["mreal"], float(rri))
  j, fi = _corner_weights(t["logmi"], np.log(max(float(iri), 1e-9)))
  w = np.array([(1 - fr) * (1 - fi), fr * (1 - fi), (1 - fr) * fi, fr * fi])
  qe = (w[0] * t["qext"][i, j] + w[1] * t["qext"][i + 1, j]
        + w[2] * t["qext"][i, j + 1] + w[3] * t["qext"][i + 1, j + 1])
  qs = (w[0] * t["qsca"][i, j] + w[1] * t["qsca"][i + 1, j]
        + w[2] * t["qsca"][i, j + 1] + w[3] * t["qsca"][i + 1, j + 1])
  return (np.interp(logx, t["logsp"], qe), np.interp(logx, t["logsp"], qs))


def Model(wvl, size_equ, dndlogdp, dpg, RRI, IRI, nonabs_fraction, shape,
          density, RH, kappa, num_theta, path_optical_dataset=None,
          path_mopsmap_executable=None):
  """
  Signature-compatible subset of mopsmap_wrapper.Model for spherical modes.

  wvl: wavelengths in nm (scalar or array). All other per-mode inputs are
  dicts keyed by mode as in the wrapper; dndlogdp in m^-3, dpg diameters in
  micrometer. RH/kappa/num_theta/density/size_equ and the two path arguments
  are accepted for compatibility and ignored (ISARA humidifies by adjusting
  dpg and the CRI itself and always calls with RH=0, kappa=0; size
  equivalence is meaningless for spheres). Raises for non-sphere shapes.
  Returns {'ext_coeff_{w}_m-1', 'ssa_{w}'} per wavelength.
  """
  t = load_table()
  wvl = np.array(wvl, ndmin=1)
  ext_tot = np.zeros(wvl.size)
  sca_tot = np.zeros(wvl.size)
  for key in dndlogdp:
    if shape[key] != "sphere":
      raise ValueError("sphere_optics.Model supports shape='sphere' only; "
                       f"mode '{key}' has shape '{shape[key]}'")
    n_bins = np.asarray(dndlogdp[key], float).reshape(-1)
    d_bins = np.asarray(dpg[key], float).reshape(-1)
    logd = np.log10(d_bins)
    n_f = max(N_FINE_MIN, N_FINE_PER_BIN * (d_bins.size - 1))
    logd_f = np.linspace(logd[0], logd[-1], n_f)
    d_f = 10.0 ** logd_f
    nn = np.interp(logd_f, logd, n_bins)
    area = np.pi / 4.0 * d_f ** 2 * nn        # um2 m-3 per logD
    for k, w in enumerate(wvl):
      logx = np.log(np.pi * d_f / (float(w) / 1000.0))
      qe, qs = _q_at(t, RRI[key], IRI[key], logx)
      # um2 m-3 -> m-1 is a factor 1e-12
      ext_tot[k] += np.trapz(qe * area, logd_f) * 1e-12
      sca_tot[k] += np.trapz(qs * area, logd_f) * 1e-12
  out = {}
  for k, w in enumerate(wvl):
    out[f"ext_coeff_{w}_m-1"] = ext_tot[k]
    out[f"ssa_{w}"] = sca_tot[k] / ext_tot[k] if ext_tot[k] > 0 else np.nan
  return out
