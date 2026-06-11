"""LUT unit tests with a stubbed (linear, analytic) MOPSMAP model.

The stub is exactly linear in the supplied dN/dlogDp, mirroring the real
MOPSMAP behavior for `distr_file dndlogr` inputs, so the LUT path and the
per-candidate subprocess path must produce identical Retr_CRI results.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ISARA  # noqa: E402
import optics_lut  # noqa: E402

WVL = [450, 470, 532, 550, 660, 700]
DPG = np.logspace(np.log10(0.02), np.log10(2.0), 10)


def fake_model(wvl, size_equ, dndlogdp, dpg, RRI, IRI, nonabs_fraction, shape,
               density, RH, kappa, num_theta, path_optical_dataset,
               path_mopsmap_executable):
    """Analytic stand-in: ext linear in sd, ssa a smooth function of IRI."""
    (mode,) = dndlogdp.keys()
    sd = np.asarray(dndlogdp[mode], float)
    d = np.asarray(dpg[mode], float)
    rri, iri = float(RRI[mode]), float(IRI[mode])
    out = {}
    for w in np.atleast_1d(wvl):
        kernel = d**2 * (rri - 1.0) * (450.0 / float(w)) * 1e-12
        ext = float(np.sum(sd * kernel))
        ssa = 1.0 / (1.0 + 60.0 * iri)
        out[f"ext_coeff_{int(w)}_m-1"] = ext
        out[f"ssa_{int(w)}"] = ssa
    return out


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(ISARA, "MMModel", fake_model)
    monkeypatch.setattr(optics_lut.mopsmap_wrapper, "Model", fake_model)


@pytest.fixture
def lut(stubbed):
    grid = ISARA.default_CRI_grid()
    wvl_sorted = np.sort(np.array(WVL, float))
    return optics_lut.build(
        wvl_sorted, grid, DPG, "ds", "exe", n_workers=1, verbose=False
    )


def make_measurements(sd_m3, rri=1.53, iri=0.005):
    """Forward-model 'truth' through the same stub."""
    res = fake_model(WVL, {"PSD": "cs"}, {"PSD": sd_m3}, {"PSD": DPG},
                     {"PSD": rri}, {"PSD": iri}, {"PSD": 0}, {"PSD": "sphere"},
                     {"PSD": 1.0}, 0, 0, 2, "", "")
    sca, ab = {}, {}
    for w in [450, 550, 700]:
        sca[w] = res[f"ssa_{w}"] * res[f"ext_coeff_{w}_m-1"]
    for w in [470, 532, 660]:
        ab[w] = res[f"ext_coeff_{w}_m-1"] - res[f"ssa_{w}"] * res[f"ext_coeff_{w}_m-1"]
    meas = {}
    for w, v in sca.items():
        meas[f"dry_meas_sca_coef_{w}_m-1"] = v
    for w, v in ab.items():
        meas[f"dry_meas_abs_coef_{w}_m-1"] = v
    return meas


def retr_cri(meas, sd_m3, lut=None):
    wvl_dict = {"sca": np.array([450, 550, 700]), "abs": np.array([470, 532, 660])}
    return ISARA.Retr_CRI(
        wvl_dict, None, meas, {"PSD": np.asarray(sd_m3, float)}, {"PSD": DPG},
        ISARA.default_CRI_grid(), {"PSD": "cs"}, {"PSD": 0}, {"PSD": "sphere"},
        {"PSD": 1.0}, 2, "ds", "exe", lut=lut,
    )


def test_lut_coefficients_match_direct(lut):
    sd = 1e6 * (100.0 + 50.0 * np.arange(len(DPG)))
    ext, sca = lut.coefficients(sd)
    grid = ISARA.default_CRI_grid()
    for i in (0, len(grid) // 2, len(grid) - 1):
        res = fake_model(np.sort(np.array(WVL)), {"PSD": "cs"}, {"PSD": sd},
                         {"PSD": DPG}, {"PSD": grid[i, 0]}, {"PSD": grid[i, 1]},
                         {"PSD": 0}, {"PSD": "sphere"}, {"PSD": 1.0}, 0, 0, 2, "", "")
        for j, w in enumerate(sorted(WVL)):
            assert np.isclose(ext[i, j], res[f"ext_coeff_{w}_m-1"], rtol=1e-12)


def test_lut_and_subprocess_paths_identical(stubbed, lut):
    sd = 1e6 * np.full(len(DPG), 500.0)
    meas = make_measurements(sd, rri=1.53, iri=0.005)
    r_no_lut = retr_cri(meas, sd, lut=None)
    r_lut = retr_cri(meas, sd, lut=lut)
    assert r_no_lut["dry_RRI_unitless"] == pytest.approx(r_lut["dry_RRI_unitless"])
    assert r_no_lut["dry_IRI_unitless"] == pytest.approx(r_lut["dry_IRI_unitless"])
    for k in r_no_lut:
        if r_no_lut[k] is not None:
            assert r_lut[k] == pytest.approx(r_no_lut[k]), k


def test_mismatched_grid_falls_back(stubbed, lut, monkeypatch):
    calls = {"n": 0}
    real = fake_model

    def counting_model(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ISARA, "MMModel", counting_model)
    sd = 1e6 * np.full(len(DPG) - 1, 500.0)  # one bin dropped -> grid mismatch

    wvl_dict = {"sca": np.array([450, 550, 700]), "abs": np.array([470, 532, 660])}
    meas = make_measurements(np.append(sd, 0.0))
    ISARA.Retr_CRI(
        wvl_dict, None, meas, {"PSD": sd}, {"PSD": DPG[:-1]},
        ISARA.default_CRI_grid(), {"PSD": "cs"}, {"PSD": 0}, {"PSD": "sphere"},
        {"PSD": 1.0}, 2, "ds", "exe", lut=lut,
    )
    # subprocess path must have run (>= one call per candidate)
    assert calls["n"] >= len(ISARA.default_CRI_grid())


def test_save_load_roundtrip(lut, tmp_path):
    p = lut.save(tmp_path / "lut_test.npz")
    lut2 = optics_lut.OpticsLUT.load(p)
    assert lut2.fingerprint == lut.fingerprint
    sd = 1e6 * np.full(len(DPG), 123.0)
    np.testing.assert_allclose(lut.coefficients(sd)[0], lut2.coefficients(sd)[0])
    assert lut2.matches(np.sort(np.array(WVL, float)), ISARA.default_CRI_grid(),
                        DPG, "cs", 0, "sphere")


def test_retr_psd_passes_lut_through(stubbed, lut):
    sd_cm3 = np.full(len(DPG), 500.0)
    meas = make_measurements(sd_cm3 * 1e6, rri=1.53, iri=0.005)
    out = ISARA.Retr_PSD(
        radii_um=DPG / 2,
        dndlogdp_cm3=sd_cm3,
        dry_sca_coef=np.array([meas[f"dry_meas_sca_coef_{w}_m-1"] for w in (450, 550, 700)]),
        dry_abs_coef=np.array([meas[f"dry_meas_abs_coef_{w}_m-1"] for w in (470, 532, 660)]),
        dry_wvl={"sca": [450, 550, 700], "abs": [470, 532, 660]},
        lut=lut,
    )
    assert out["attempt_flag_CRI_unitless"] == 2
    assert np.isclose(out["dry_RRI_unitless"], 1.53, atol=0.011)
