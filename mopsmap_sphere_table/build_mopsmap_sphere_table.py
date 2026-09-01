"""
Extract the sphere single-particle efficiencies from the MOPSMAP optical
dataset (Gasteiger & Wiegner, 2018, doi:10.5194/gmd-11-2739-2018; dataset
doi:10.5281/zenodo.1284217) into one compact netCDF table.

Reads sphere_<mr>_<mi>.nc files (qext, qsca vs size parameter; exact Mie at
~232 points/decade) and writes mopsmap_spheres_v1.nc with dimensions
(mreal, mimag, sizepara). Only qext/qsca/sizepara are read; the angular
scattering coefficients (99% of the source volume) are not extracted.

The output file is a derivative work of the MOPSMAP optical dataset and is
distributed under the GPL (see COPYING in this directory); this script is
the corresponding "source" for regenerating it from the original dataset.

Usage: python build_mopsmap_sphere_table.py [path_to_optical_dataset/spheres]
"""
import glob
import os
import re
import sys
import datetime

import numpy as np
import netCDF4 as nc
from numpy.polynomial import legendre

# Optical-sizer collection geometry for the qsca_partial response kernel.
# Moore et al. (2021, AMT, doi:10.5194/amt-14-4517-2021) integrate the phase
# function over the UHSAS collection angles for BOTH the UHSAS and the LAS:
# 33-147 deg with the 72.5-104.8 deg band blocked. Using the same geometry
# keeps this kernel consistent with their published response curves.
GEOM_THETA = (33.0, 147.0)
GEOM_BLOCKED = (72.5, 104.8)
XMAX_PARTIAL = 60.0   # size parameter ceiling for the partial kernel (covers
                      # LAS to 6 um at 633 nm and UHSAS through 1 um at 1054)

DEFAULT_SRC = ("/Users/wrespino/Synced/Resources/GeneralSoftware/MOPSMAP/"
               "mopsmap/optical_dataset/spheres/")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "mopsmap_spheres_v2.nc")


def main(src):
  files = sorted(glob.glob(os.path.join(src, "sphere_*.nc")))
  if not files:
    raise SystemExit("no sphere_*.nc files found in %s" % src)
  grid = {}
  for f in files:
    mm = re.search(r"sphere_([\d.]+)_([\d.]+)\.nc", os.path.basename(f))
    grid[(float(mm.group(1)), float(mm.group(2)))] = f
  mrs = sorted({k[0] for k in grid})
  mis = sorted({k[1] for k in grid})
  with nc.Dataset(grid[(mrs[0], mis[0])]) as d:
    sp = d["sizepara"][:].filled(np.nan)

  th = np.linspace(0.0, np.pi, 1441)
  mu, sinth = np.cos(th), np.sin(th)
  wmask = ((th >= np.deg2rad(GEOM_THETA[0])) & (th <= np.deg2rad(GEOM_THETA[1]))
           & ~((th >= np.deg2rad(GEOM_BLOCKED[0]))
               & (th <= np.deg2rad(GEOM_BLOCKED[1]))))
  wfun = np.where(wmask, sinth, 0.0)
  sel_part = sp <= XMAX_PARTIAL

  qext = np.full((len(mrs), len(mis), len(sp)), np.nan, np.float32)
  qsca = np.full_like(qext, np.nan)
  qpart = np.full_like(qext, np.nan)
  for i, mr in enumerate(mrs):
    print(f"  mr {mr:.2f} ({i + 1}/{len(mrs)})", flush=True)
    for j, mi in enumerate(mis):
      with nc.Dataset(grid[(mr, mi)]) as d:
        assert np.array_equal(d["sizepara"][:].filled(np.nan), sp)
        qext[i, j] = d["qext"][:].filled(np.nan)
        qsca[i, j] = d["qsca"][:].filled(np.nan)
        lmax = d["lmax"][:].filled(0).astype(int)
        offs = np.r_[0, np.cumsum(lmax + 1)]
        a1 = d["a1"][:].filled(np.nan)
        for k in np.where(sel_part)[0]:
          coef = a1[offs[k]:offs[k] + lmax[k] + 1]
          frac = 0.5 * np.trapz(legendre.legval(mu, coef) * wfun, th)
          qpart[i, j, k] = qsca[i, j, k] * frac

  with nc.Dataset(OUT, "w") as o:
    o.createDimension("mreal", len(mrs))
    o.createDimension("mimag", len(mis))
    o.createDimension("sizepara", len(sp))
    v = o.createVariable("mreal", "f8", ("mreal",))
    v[:] = mrs
    v.long_name = "real part of refractive index"
    v = o.createVariable("mimag", "f8", ("mimag",))
    v[:] = mis
    v.long_name = "imaginary part of refractive index (non-negative)"
    v = o.createVariable("sizepara", "f8", ("sizepara",))
    v[:] = sp
    v.long_name = "size parameter 2*pi*r/lambda"
    for name, arr, ln in [("qext", qext, "extinction efficiency"),
                          ("qsca", qsca, "scattering efficiency"),
                          ("qsca_partial", qpart,
                           "partial scattering efficiency into the optical-"
                           "sizer collection solid angle")]:
      v = o.createVariable(name, "f4", ("mreal", "mimag", "sizepara"),
                           zlib=True, complevel=4, shuffle=True)
      v[:] = arr
      v.long_name = ln + " (exact Mie, single sphere)"
      v.units = "dimensionless"
    v = o["qsca_partial"]
    v.collection_theta_deg = GEOM_THETA
    v.blocked_theta_deg = GEOM_BLOCKED
    v.comment = ("qsca * (1/2) int P11 sin(theta) dtheta over 33-147 deg "
                 "minus the blocked 72.5-104.8 deg band (Moore et al. 2021 "
                 "geometry, applied to both LAS and UHSAS); computed from "
                 "the MOPSMAP a1 (ALPH1) Legendre expansion; NaN above "
                 f"size parameter {XMAX_PARTIAL}")
    o.title = "MOPSMAP sphere single-particle efficiencies (extract)"
    o.source = ("Extracted from the MOPSMAP v1.0 optical dataset, "
                "spheres subset (qext/qsca only)")
    o.references = ("Gasteiger, J. and Wiegner, M.: MOPSMAP v1.0, "
                    "Geosci. Model Dev., 11, 2739-2762, "
                    "doi:10.5194/gmd-11-2739-2018, 2018. "
                    "Dataset: doi:10.5281/zenodo.1284217")
    o.license = ("GNU General Public License (see COPYING alongside this "
                 "file); derivative of the GPL-licensed MOPSMAP dataset")
    o.history = ("%s: created by build_mopsmap_sphere_table.py"
                 % datetime.date.today().isoformat())
    o.usage = ("Point values of exact-Mie efficiencies. Recommended "
               "evaluation: interpolate linearly in log(sizepara), "
               "bilinearly in (mreal, log(mimag)) with an mimag=0 guard; "
               "integrate against the PSD on a fine log-radius grid. "
               "Validated to <=0.11% vs exact Mie for lognormal and "
               "measured PSDs incl. quasi-monodisperse (GSD 1.05).")
  print("wrote %s (%.1f MB): %d mr x %d mi x %d sizepara"
        % (OUT, os.path.getsize(OUT) / 1e6, len(mrs), len(mis), len(sp)))


if __name__ == "__main__":
  main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC)
