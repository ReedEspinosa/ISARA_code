# MOPSMAP sphere efficiency table (extract)

`mopsmap_spheres_v1.nc` (1.5 MB) holds exact-Mie single-sphere extinction
and scattering efficiencies extracted from the **MOPSMAP v1.0 optical
dataset** — 10 `mreal` (1.28–1.64) × 17 `mimag` (0–0.1376) × 2085 size
parameters (1e-6–1013, ~232 points/decade). Only `qext`/`qsca` were
extracted; the angular-scattering content of the source dataset (99% of
its 1.9 GB) is not included. Regenerate with
`python build_mopsmap_sphere_table.py <path-to-optical_dataset/spheres>`.

## License and attribution (read this before redistributing)

This netCDF file is a **derivative work of the MOPSMAP optical dataset**
by Josef Gasteiger and Matthias Wiegner and is distributed under the
**GNU General Public License** (full text in `COPYING` here; the Zenodo
record lists GPL v2, the distributed bundle ships the v3 text). It is
included in this repository as *mere aggregation*: the MIT license at the
repository root applies to the ISARA code, **not** to this data file or
anything derived from it. The extraction script in this directory is the
corresponding source for regenerating the file from the original dataset.

Original dataset and program: <https://doi.org/10.5281/zenodo.1284217>.
Publications using results derived from this table are requested to cite:

> Gasteiger, J. and Wiegner, M.: MOPSMAP v1.0: a versatile tool for
> modeling of aerosol optical properties, Geosci. Model Dev., 11,
> 2739–2762, https://doi.org/10.5194/gmd-11-2739-2018, 2018.

## Usage conventions (validated)

Values are **point samples** of exact Mie efficiencies (unlike
basis-averaged GRASP kernel tables — no de-smoothing step is needed).
Recommended evaluation, validated against direct high-resolution Mie
(worst case 0.21% over lognormal PSDs from GSD 1.05 to 2.0, off-node
CRIs, 450–700 nm; see ASCENT-ACP `scripts/mie_ground_truth.py`):

1. bilinear interpolation in (`mreal`, log `mimag`), guarding `mimag=0`
   with a floor of ~1e-9;
2. linear interpolation in log(`sizepara`);
3. integrate against the PSD on a fine log-radius grid (never sample the
   PSD at the table nodes).

The 0.04 `mreal` spacing contributes ≤0.11% even mid-cell; the dense
size-parameter grid keeps quasi-monodisperse PSDs (GSD ≥ 1.05) to ≤0.03%
where basis-averaged 41-node tables err by up to ~3.5%.
