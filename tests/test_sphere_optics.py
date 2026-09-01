"""Table forward engine (sphere_optics) vs physical sanity + wrapper parity."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sphere_optics  # noqa: E402

WVL = np.array([450, 550, 700])


def _args(rri=1.53, iri=0.007):
  dpg = np.logspace(np.log10(0.02), np.log10(1.0), 51)
  sd = 1e9 * np.exp(-0.5 * ((np.log(dpg) - np.log(0.2)) / 0.7) ** 2)  # m^-3
  return dict(size_equ={'m': 'cs'}, dndlogdp={'m': sd}, dpg={'m': dpg},
              RRI={'m': rri}, IRI={'m': iri}, nonabs_fraction={'m': 0},
              shape={'m': 'sphere'}, density={'m': 1.0}, RH=0, kappa=0,
              num_theta=2)


def test_physical_sanity_and_linearity():
  r = sphere_optics.Model(WVL, **_args())
  for w in WVL:
    ext = r[f'ext_coeff_{w}_m-1']
    ssa = r[f'ssa_{w}']
    assert 1e-6 < ext < 1e-3          # tens-hundreds of Mm-1 for this PSD
    assert 0.8 < ssa < 1.0
  # coefficients are linear in number concentration
  a = _args()
  a['dndlogdp'] = {'m': a['dndlogdp']['m'] * 2.0}
  r2 = sphere_optics.Model(WVL, **a)
  assert r2['ext_coeff_550_m-1'] == pytest.approx(2 * r['ext_coeff_550_m-1'], rel=1e-9)
  assert r2['ssa_550'] == pytest.approx(r['ssa_550'], rel=1e-9)


def test_iri_zero_gives_ssa_one():
  r = sphere_optics.Model(WVL, **_args(iri=0.0))
  assert r['ssa_550'] == pytest.approx(1.0, abs=2e-4)


def test_non_sphere_rejected():
  a = _args()
  a['shape'] = {'m': 'spheroid oblate 1.7'}
  with pytest.raises(ValueError, match="sphere"):
    sphere_optics.Model(WVL, **a)


def test_matches_mopsmap_executable():
  """Parity with the Fortran engine (skipped when it is not installed)."""
  exe = "/Users/wrespino/Synced/Resources/GeneralSoftware/MOPSMAP/mopsmap/mopsmap"
  dat = "/Users/wrespino/Synced/Resources/GeneralSoftware/MOPSMAP/mopsmap/optical_dataset/"
  if not os.path.exists(exe):
    pytest.skip("MOPSMAP executable not available")
  import mopsmap_wrapper
  rt = sphere_optics.Model(WVL, **_args())
  rm = mopsmap_wrapper.Model(WVL, **_args(), path_optical_dataset=dat,
                             path_mopsmap_executable=exe)
  for w in WVL:
    assert rt[f'ext_coeff_{w}_m-1'] == pytest.approx(rm[f'ext_coeff_{w}_m-1'], rel=5e-3)
    assert rt[f'ssa_{w}'] == pytest.approx(rm[f'ssa_{w}'], rel=2e-3)


def _retr_kwargs(**over):
  dpg = np.logspace(np.log10(0.02), np.log10(1.0), 41)
  dnd = 800 * np.exp(-0.5 * ((np.log(dpg) - np.log(0.2)) / 0.6) ** 2)  # cm^-3
  kw = dict(radii_um=dpg / 2, dndlogdp_cm3=dnd,
            dry_wvl={'sca': [450, 550, 700], 'abs': [470, 532, 660]},
            forward_engine='table')
  kw.update(over)
  return kw, dpg, dnd


def _forward_truth(dpg, dnd, rri, iri, kappa=None, rh=80.0):
  """Table-engine coefficients for a known CRI (and optional wet state)."""
  args = dict(size_equ={'m': 'cs'}, dndlogdp={'m': dnd * 1e6},
              nonabs_fraction={'m': 0}, shape={'m': 'sphere'},
              density={'m': 1.0}, RH=0, kappa=0, num_theta=2)
  if kappa is None:
    d, rr, ii = {'m': dpg}, rri, iri
  else:
    gf = (1 + kappa * rh / (100 - rh)) ** (1 / 3)
    d = {'m': dpg * gf}
    rr = (rri + (gf ** 3 - 1) * 1.33) / gf ** 3
    ii = iri / gf ** 3
  r = sphere_optics.Model(np.array([450, 470, 532, 550, 660, 700]),
                          dpg=d, RRI={'m': rr}, IRI={'m': ii}, **args)
  sca = np.array([r[f'ssa_{w}'] * r[f'ext_coeff_{w}_m-1'] for w in (450, 550, 700)])
  ab = np.array([(1 - r[f'ssa_{w}']) * r[f'ext_coeff_{w}_m-1'] for w in (470, 532, 660)])
  return sca, ab


def test_chi2_wmean_recovers_truth_cri_and_kappa():
  import ISARA
  kw, dpg, dnd = _retr_kwargs(estimator='chi2-wmean')
  rri_t, iri_t, kap_t = 1.52, 0.005, 0.30
  sca, ab = _forward_truth(dpg, dnd, rri_t, iri_t)
  wet, _ = _forward_truth(dpg, dnd, rri_t, iri_t, kappa=kap_t)
  out = ISARA.Retr_PSD(dry_sca_coef=sca, dry_abs_coef=ab,
                       wet_sca_coef=wet[1:2], wet_wvl={'sca': [550]},
                       CRI_p=ISARA.default_CRI_grid(1.47, 1.56), **kw)
  assert out['attempt_flag_CRI_unitless'] == 2
  assert out['dry_RRI_unitless'] == pytest.approx(rri_t, abs=0.005)
  assert out['dry_IRI_unitless'] == pytest.approx(iri_t, abs=0.001)
  assert out['dry_CRI_min_chi2_unitless'] < 0.05
  assert out['attempt_flag_kappa_unitless'] == 2
  assert out['kappa_unitless'] == pytest.approx(kap_t, abs=0.01)
  assert out['kappa_min_chi2_unitless'] < 0.1
  assert out['kappa_std_unitless'] < 0.05


def test_chi2_wmean_gate_fails_inconsistent_measurements():
  import ISARA
  kw, dpg, dnd = _retr_kwargs(estimator='chi2-wmean')
  sca, ab = _forward_truth(dpg, dnd, 1.52, 0.005)
  out = ISARA.Retr_PSD(dry_sca_coef=sca * np.array([3.0, 1.0, 0.3]),
                       dry_abs_coef=ab,
                       CRI_p=ISARA.default_CRI_grid(1.47, 1.56), **kw)
  assert out['attempt_flag_CRI_unitless'] == 1  # attempted, no solution
  assert np.isnan(out['dry_RRI_unitless'])


def test_linf_estimator_still_works_on_table_engine():
  import ISARA
  kw, dpg, dnd = _retr_kwargs(estimator='linf-mean')
  sca, ab = _forward_truth(dpg, dnd, 1.52, 0.005)
  out = ISARA.Retr_PSD(dry_sca_coef=sca, dry_abs_coef=ab,
                       CRI_p=ISARA.default_CRI_grid(1.47, 1.56), **kw)
  assert out['attempt_flag_CRI_unitless'] == 2
  assert abs(out['dry_RRI_unitless'] - 1.52) < 0.02


def test_obs_cov_forgives_patterned_residual_only():
  """Full-covariance chi2: a residual along a nuisance direction passes,
  the same-size residual in an inconsistent pattern fails."""
  import ISARA
  kw, dpg, dnd = _retr_kwargs(estimator='chi2-wmean')
  sca, ab = _forward_truth(dpg, dnd, 1.52, 0.005)
  meas_sig = np.r_[0.05 * sca, 0.3e-6 * np.ones(3)]
  # nuisance direction: common multiplicative shift of all sca channels
  dy = np.r_[0.25 * sca, np.zeros(3)]
  S = np.diag(meas_sig ** 2) + np.outer(dy, dy)
  patterned = sca * 1.22          # ~1 sigma along the nuisance direction
  odd = sca * np.array([1.22, 1.0, 0.82])   # same size, wrong shape
  ok = ISARA.Retr_PSD(dry_sca_coef=patterned, dry_abs_coef=ab,
                      CRI_p=ISARA.default_CRI_grid(1.47, 1.56),
                      obs_cov=S, **kw)
  bad = ISARA.Retr_PSD(dry_sca_coef=odd, dry_abs_coef=ab,
                       CRI_p=ISARA.default_CRI_grid(1.47, 1.56),
                       obs_cov=S, **kw)
  assert ok['attempt_flag_CRI_unitless'] == 2
  assert bad['dry_CRI_min_chi2_unitless'] > ok['dry_CRI_min_chi2_unitless'] * 3
