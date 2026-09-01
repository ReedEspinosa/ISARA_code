# ISARA retrieval: theory basis (as implemented, 2026-08-31)

This documents the algorithm as it stands in this repository. The
campaign-pipeline layer (QC, windowing, PSD construction, impactor
weighting, uncertainty propagation, netCDF output) lives in the sibling
ASCENT-ACP repo (`THEORY.md` there); this document covers the
single-retrieval core: `Retr_PSD` and the forward engines.

## 1. Problem statement

Given, for one time window:
- a dry particle size distribution dN/dlogDp on arbitrary bin-center
  diameters (any units-consistent grid; NaN bins are dropped),
- dry scattering coefficients at 3 wavelengths and dry absorption
  coefficients at 3 wavelengths (m^-1),
- optionally a humidified scattering coefficient (m^-1) at a known RH,

retrieve a spectrally FLAT dry complex refractive index (RRI, IRI), and
optionally the kappa-Kohler bulk hygroscopicity, then forward-calculate
the full optical set (sca/abs/ext/SSA) for the dry, fixed-RH ("wet") and
ambient-RH states at any requested wavelengths.

Assumptions inherited from the ISARA heritage (Schlosser et al., 2025):
homogeneous spheres, one wavelength-independent CRI, one bulk kappa
applied to all sizes, volume mixing with water (1.33 + 0i) for humidified
CRI: RRI_w = (RRI_d + (gf^3-1)*1.33)/gf^3, gf = (1 + kappa*RH/(100-RH))^(1/3).

## 2. Forward model (engines)

`forward_engine` selects between two physically identical Mie forward
models; retrieval logic is engine-independent.

- **`table` (default in ASCENT-ACP):** `sphere_optics.py` integrates the
  single-sphere efficiency table `mopsmap_sphere_table/mopsmap_spheres_v1.nc`
  (extracted from the MOPSMAP v1.0 optical dataset, Gasteiger & Wiegner
  2018; 10 mreal 1.28-1.64 x 17 mimag 0-0.1376 x 2085 size parameters
  ~232 pts/decade, exact Mie point values). Evaluation: bilinear
  interpolation in (mreal, log mimag), linear in log size parameter, and
  integration of piecewise-linear dN/dlogDp between bin centers (zero
  outside) on a fine log-diameter quadrature grid — the same PSD
  representation MOPSMAP itself uses. Validated against direct
  high-resolution Mie to <= 0.21% worst case (PSDs GSD 1.05-2.0, off-node
  CRIs, 450-700 nm) and against the MOPSMAP executable to ~0.1%.
  ~0.4 ms per call. See mopsmap_sphere_table/README.md for provenance,
  GPL licensing of the data file, and the regeneration script.
- **`mopsmap`:** the historical Fortran subprocess (`mopsmap_wrapper.py`),
  ~33 ms per call; retained as the validation reference. The `optics_lut`
  hat-kernel machinery accelerates only this engine's CRI search.

Engine A/B on real windows: identical success flags; |dRRI| <= 0.0036,
|dkappa| <= 0.002 — far below retrieval uncertainty.

Notes on why the table is trustworthy: point values of exact Mie at
~232 pts/decade resolve the resonance structure that broad PSDs then
average; a 0.04 mreal grid contributes <= 0.11% even mid-cell because
ripple misalignment between nodes integrates out against any realistic
PSD width. (Basis-averaged 41-node GRASP kernels were evaluated as an
alternative and are exact after mass-matrix de-smoothing for broad PSDs,
but degrade for quasi-monodisperse PSDs; full analysis in ASCENT-ACP
`GRASP_KERNEL_PLAN.md`.)

## 3. Dry CRI grid search (`Retr_CRI`)

Candidates: the Cartesian grid RRI in [1.47, 1.56] step 0.01 (config) x
IRI in {0, 1e-7..1e-4 decades, 0.001..0.030 step 0.001} — 350 pairs.
For every candidate the forward model predicts the 3+3 dry channels.

**Misfit.** Three nested options (`estimator` / sigma inputs):

1. Legacy tolerances: candidate "accepted" if ALL sca channels within
   20% relative AND all abs channels within 1 Mm^-1 (an L-infinity gate).
2. Diagonal instrument sigmas (`sca_sigma`, `abs_sigma`): reduced
   chi^2 = (1/6) sum_i (r_i/sigma_i)^2 with per-window, per-channel
   1-sigma values.
3. Full covariance (`obs_cov`, the ASCENT-ACP default): generalized
   chi^2 = r' S^-1 r / 6 where S = Sigma_meas + sum_k dy_k dy_k'. The
   rank-1 terms are the coefficient signatures of correlated model
   nuisances (PSD diameter-scale, concentration scale, impactor
   parameters) and correlated measurement modes. This MARGINALIZES the
   fit over known model uncertainty: a residual pattern lying along a
   nuisance direction is forgiven; an inconsistent spectral shape of the
   same size still fails.

**Solution.** `estimator='chi2-wmean'` (default in ASCENT-ACP): the
Gaussian-posterior weighted mean over the grid, weights
w_i = exp(-1/2 chi2_tot,i), success gate min reduced chi^2 <= 1;
reported diagnostics: min chi^2, n(chi^2<=1), posterior-weighted stds of
RRI and IRI. `estimator='linf-mean'` reproduces the historical mean of
L-infinity-accepted candidates (re-verified at the mean). The estimator
study (ASCENT-ACP `scripts/estimator_study.py`) showed the weighted mean
has the best RMSE and smallest bias of the candidates considered, and
its continuous weights remove the acceptance-boundary fragility of
binary gates.

## 4. Kappa retrieval (`Retr_kappa`)

Scan kappa in [0, 1.4) step 0.001. For each candidate: grow all bins by
gf, mix the CRI with water by volume, forward-model the humidified
scattering channel(s). Selection mirrors the CRI stage: historical =
first kappa with all channels within 1% (ascending scan; biases low);
`chi2-wmean` = posterior mean with sigma = `wet_sigma` (or the 1%
legacy), gate min reduced chi^2 <= 1, reporting kappa_min_chi2 and the
posterior std. Because the ASCENT-ACP fitting target is SYNTHESIZED from
the same dry nephelometer channel (dry Sc550 gamma-adjusted), instrument
calibration cancels in the ratio; the appropriate wet sigma is the
gamma-parameterization uncertainty (~1%) plus the non-cancelling noise
floor — not the full instrument model.

## 5. Humidified states (`humidified_optics`)

For retrieved (CRI, kappa) and any RH < 100: grow diameters by gf, mix
CRI with water, forward-calculate sca/abs/ext/SSA at all requested
wavelengths. Humidified absorption is model-derived (absorption is only
measured dry). The ambient state uses the window-mean ambient RH subject
to the pipeline's RH ceiling policy.

## 6. Known limitations (documented, not yet addressed here)

- Spectrally flat CRI: high-AAE absorption spectra (e.g. brown carbon,
  or PSAP spectral artifacts) cannot be fit; they fail the gate.
- LAS diameters are dry AmmSO4-optical; a particle RI different from the
  calibration produces a correlated sizing error. In ACTIVATE 2021 this
  appears as a systematic ~+4-5% preferred diameter-scale shift
  (`sizing_scale_shift` diagnostic in ASCENT-ACP V9). The planned
  raw-signal refit (fit the LAS response, not the calibrated PSD) removes
  this circularity; see ASCENT-ACP todo_reed.txt (HIGH PRIORITY).
- Single bulk kappa and sphericity weaken for coarse-mode-influenced
  windows (see Retr_PSD docstring notes).
