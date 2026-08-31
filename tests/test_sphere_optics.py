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
