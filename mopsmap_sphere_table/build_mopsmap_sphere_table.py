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

DEFAULT_SRC = ("/Users/wrespino/Synced/Resources/GeneralSoftware/MOPSMAP/"
               "mopsmap/optical_dataset/spheres/")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "mopsmap_spheres_v1.nc")


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

  qext = np.full((len(mrs), len(mis), len(sp)), np.nan, np.float32)
  qsca = np.full_like(qext, np.nan)
  for i, mr in enumerate(mrs):
    for j, mi in enumerate(mis):
      with nc.Dataset(grid[(mr, mi)]) as d:
        assert np.array_equal(d["sizepara"][:].filled(np.nan), sp)
        qext[i, j] = d["qext"][:].filled(np.nan)
        qsca[i, j] = d["qsca"][:].filled(np.nan)

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
                          ("qsca", qsca, "scattering efficiency")]:
      v = o.createVariable(name, "f4", ("mreal", "mimag", "sizepara"),
                           zlib=True, complevel=4, shuffle=True)
      v[:] = arr
      v.long_name = ln + " (exact Mie, single sphere)"
      v.units = "dimensionless"
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
