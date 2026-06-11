"""Unit tests for the issue-#5 size-distribution merge/overlap helpers and the
direct-PSD entry point ISARA.Retr_PSD.

These tests need neither the MOPSMAP executable nor the optical dataset:
mopsmap_wrapper.Model is replaced with a stub, and only the pure-python
plumbing (overlap trimming, sorting, label mapping, unit conversions) is
verified. Run with either:
    python tests/test_psd_merge.py
    python -m pytest tests/test_psd_merge.py
"""

import io
import os
import sys
import types
import contextlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ISARA_Data_Retrieval imports pathos at module level; stub it so the pure
# helper functions can be imported on machines without pathos installed.
if "pathos" not in sys.modules:
    _pathos = types.ModuleType("pathos")
    _pathos_mp = types.ModuleType("pathos.multiprocessing")

    class _StubProcessPool:
        def __init__(self, *args, **kwargs):
            pass

    _pathos_mp.ProcessPool = _StubProcessPool
    _pathos.multiprocessing = _pathos_mp
    sys.modules["pathos"] = _pathos
    sys.modules["pathos.multiprocessing"] = _pathos_mp

import ISARA
import ISARA_Data_Retrieval as IDR


def _example_bins():
    """SMPS overlapping UHSAS: SMPS dpg 0.113 and 0.226 lie inside UHSAS coverage."""
    dpg = {
        "SMPS": np.array([0.014, 0.028, 0.057, 0.113, 0.226]),
        "UHSAS": np.array([0.141, 0.283, 0.566]),
    }
    dpl = {
        "SMPS": np.array([0.010, 0.020, 0.040, 0.080, 0.160]),
        "UHSAS": np.array([0.100, 0.200, 0.400]),
    }
    dpu = {
        "SMPS": np.array([0.020, 0.040, 0.080, 0.160, 0.320]),
        "UHSAS": np.array([0.200, 0.400, 0.800]),
    }
    retained = {
        "SMPS": np.arange(5),
        "UHSAS": np.arange(3),
    }
    return dpg, dpl, dpu, retained


def test_overlap_rules_from_priority():
    rules = IDR.overlap_rules_from_priority(["UHSAS", "APS", "SMPS"])
    assert rules == [("UHSAS", "APS"), ("UHSAS", "SMPS"), ("APS", "SMPS")]
    assert IDR.overlap_rules_from_priority(["LAS"]) == []


def test_resolve_overlap_uhsas_wins_over_smps():
    dpg, dpl, dpu, retained = _example_bins()
    out = IDR.resolve_overlap(retained, dpg, dpl, dpu, IDR.DEFAULT_OVERLAP_RULES)
    # SMPS bins with dpg inside UHSAS coverage [0.100, 0.800] are dropped
    assert list(out["SMPS"]) == [0, 1, 2]
    # the winner is untouched
    assert list(out["UHSAS"]) == [0, 1, 2]


def test_resolve_overlap_las_wins_over_smps():
    dpg, dpl, dpu, retained = _example_bins()
    dpg["LAS"] = dpg.pop("UHSAS")
    dpl["LAS"] = dpl.pop("UHSAS")
    dpu["LAS"] = dpu.pop("UHSAS")
    retained["LAS"] = retained.pop("UHSAS")
    out = IDR.resolve_overlap(retained, dpg, dpl, dpu, IDR.DEFAULT_OVERLAP_RULES)
    assert list(out["SMPS"]) == [0, 1, 2]
    assert list(out["LAS"]) == [0, 1, 2]


def test_resolve_overlap_unrelated_pair_untouched():
    dpg, dpl, dpu, retained = _example_bins()
    # rename UHSAS to APS: no default rule relates APS and SMPS
    dpg["APS"] = dpg.pop("UHSAS")
    dpl["APS"] = dpl.pop("UHSAS")
    dpu["APS"] = dpu.pop("UHSAS")
    retained["APS"] = retained.pop("UHSAS")
    out = IDR.resolve_overlap(retained, dpg, dpl, dpu, IDR.DEFAULT_OVERLAP_RULES)
    assert list(out["SMPS"]) == [0, 1, 2, 3, 4]
    assert list(out["APS"]) == [0, 1, 2]


def test_resolve_overlap_empty_winner_is_noop():
    dpg, dpl, dpu, retained = _example_bins()
    retained["UHSAS"] = np.array([], dtype=int)
    out = IDR.resolve_overlap(retained, dpg, dpl, dpu, IDR.DEFAULT_OVERLAP_RULES)
    assert list(out["SMPS"]) == [0, 1, 2, 3, 4]


def test_merge_bins_sorted_and_labeled():
    dpg, dpl, dpu, retained = _example_bins()
    retained = IDR.resolve_overlap(retained, dpg, dpl, dpu, IDR.DEFAULT_OVERLAP_RULES)
    rank = {"UHSAS": 0, "SMPS": 1}
    sd = {
        "SMPS": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        "UHSAS": np.array([1.0, 2.0, 3.0]),
    }
    merged = IDR.merge_bins(retained, dpg, dpl, dpu, rank, sd=sd)
    assert np.all(np.diff(merged["dpg"]) > 0)
    assert np.allclose(merged["dpg"], [0.014, 0.028, 0.057, 0.141, 0.283, 0.566])
    assert list(merged["inst"]) == ["SMPS", "SMPS", "SMPS", "UHSAS", "UHSAS", "UHSAS"]
    assert list(merged["idx"]) == [0, 1, 2, 0, 1, 2]
    assert np.allclose(merged["sd"], [10.0, 20.0, 30.0, 1.0, 2.0, 3.0])
    # labels point back to the source arrays
    for j in range(len(merged["dpg"])):
        assert merged["dpg"][j] == dpg[merged["inst"][j]][merged["idx"][j]]


def test_merge_bins_duplicate_dpg_keeps_higher_priority():
    dpg = {"UHSAS": np.array([0.141, 0.283]), "SMPS": np.array([0.141])}
    dpl = {"UHSAS": np.array([0.100, 0.200]), "SMPS": np.array([0.120])}
    dpu = {"UHSAS": np.array([0.200, 0.400]), "SMPS": np.array([0.165])}
    retained = {"UHSAS": np.array([0, 1]), "SMPS": np.array([0])}
    rank = {"UHSAS": 0, "SMPS": 1}
    merged = IDR.merge_bins(retained, dpg, dpl, dpu, rank)
    assert len(merged["dpg"]) == 2
    assert list(merged["inst"]) == ["UHSAS", "UHSAS"]
    # reversed priority keeps the SMPS bin instead
    merged2 = IDR.merge_bins(retained, dpg, dpl, dpu, {"SMPS": 0, "UHSAS": 1})
    assert list(merged2["inst"]) == ["SMPS", "UHSAS"]
    assert np.all(np.diff(merged2["dpg"]) > 0)


def test_merge_bins_empty():
    merged = IDR.merge_bins({}, {}, {}, {}, {})
    assert len(merged["dpg"]) == 0
    merged = IDR.merge_bins({"SMPS": np.array([], dtype=int)},
                            {"SMPS": np.array([])}, {"SMPS": np.array([])},
                            {"SMPS": np.array([])}, {"SMPS": 0})
    assert len(merged["dpg"]) == 0
    assert "sd" not in merged


def test_grid_slot_row_mapping_with_nan_bin():
    """Per-row bins (with one NaN dropped) land in the correct standardized slots."""
    dpg, dpl, dpu, retained = _example_bins()
    retained = IDR.resolve_overlap(retained, dpg, dpl, dpu, IDR.DEFAULT_OVERLAP_RULES)
    rank = {"UHSAS": 0, "SMPS": 1}
    full_dp = IDR.merge_bins(retained, dpg, dpl, dpu, rank)
    grid_slot = {(full_dp["inst"][i], full_dp["idx"][i]): i for i in range(len(full_dp["dpg"]))}

    # row: SMPS bin 1 is NaN -> dropped from the row's retained set
    dndlogdp = {
        "SMPS": np.array([10.0, np.nan, 30.0, 40.0, 50.0]),
        "UHSAS": np.array([1.0, 2.0, 3.0]),
    }
    row_retained = {}
    for imode in retained:
        nom = retained[imode]
        row_retained[imode] = nom[np.logical_not(np.isnan(dndlogdp[imode][nom]))]
    merged = IDR.merge_bins(row_retained, dpg, dpl, dpu, rank, sd=dndlogdp)

    full_sd = np.full(len(full_dp["dpg"]), np.nan)
    for j in range(len(merged["dpg"])):
        slot = grid_slot.get((merged["inst"][j], merged["idx"][j]))
        assert slot is not None
        full_sd[slot] = merged["sd"][j]
    assert np.isnan(full_sd[1])  # the NaN SMPS bin's slot stays NaN
    assert np.allclose(full_sd[[0, 2, 3, 4, 5]], [10.0, 30.0, 1.0, 2.0, 3.0])


def _stub_model(calls):
    """Stand-in for mopsmap_wrapper.Model: constant ext/ssa at every wavelength."""
    def model(wvl, size_equ, sd, dpg, RRI, IRI, nonabs_fraction, shape, rho,
              RH, kappa, num_theta, path_optical_dataset, path_mopsmap_executable):
        calls.append({"sd": {k: np.array(v) for k, v in sd.items()},
                      "dpg": {k: np.array(v) for k, v in dpg.items()}})
        results = {}
        for w in np.array(wvl, ndmin=1):
            results[f'ext_coeff_{w}_m-1'] = 2.0e-5
            results[f'ssa_{w}'] = 0.5
        return results
    return model


def test_retr_psd_conversions_and_success():
    calls = []
    saved = ISARA.MMModel
    ISARA.MMModel = _stub_model(calls)
    try:
        radii = np.array([0.05, 0.02, 0.10, 0.30, np.nan])  # unsorted with one NaN
        dndlogdp = np.array([200.0, 100.0, 50.0, 10.0, 5.0])
        # stub gives scat = ssa*ext = 1e-5 and abs = ext - scat = 1e-5 at all wvls
        out = ISARA.Retr_PSD(
            radii, dndlogdp,
            dry_sca_coef=[1.0e-5, 1.0e-5, 1.0e-5],
            dry_abs_coef=[1.0e-5, 1.0e-5, 1.0e-5],
            dry_wvl={"sca": [450, 550, 700], "abs": [465, 520, 660]},
            wet_sca_coef=[1.0e-5], wet_wvl={"sca": [550]},
            CRI_p=np.array([[1.50, 0.01], [1.60, 0.03]]),
            kappa_p=np.array([0.20, 0.40]),
        )
    finally:
        ISARA.MMModel = saved

    # PSD plumbing: NaN dropped, sorted, radius->diameter (x2), cm-3 -> m-3 (x1e6)
    dpg_seen = calls[0]["dpg"]["PSD"]
    sd_seen = calls[0]["sd"]["PSD"]
    assert np.allclose(dpg_seen, [0.04, 0.10, 0.20, 0.60])
    assert np.allclose(sd_seen, [100.0e6, 200.0e6, 50.0e6, 10.0e6])

    # both grid points match the stub -> mean CRI; first kappa accepted
    assert out['attempt_flag_CRI_unitless'] == 2
    assert np.isclose(out["dry_RRI_unitless"], 1.55)
    assert np.isclose(out["dry_IRI_unitless"], 0.02)
    assert out['attempt_flag_kappa_unitless'] == 2
    assert np.isclose(out["kappa_unitless"], 0.20)
    assert np.isclose(out['dry_cal_sca_coef_550_m-1'], 1.0e-5)
    assert np.isclose(out['wet_cal_sca_coef_550_m-1'], 1.0e-5)


def test_retr_psd_failure_is_nan_not_none():
    calls = []
    saved = ISARA.MMModel
    ISARA.MMModel = _stub_model(calls)
    try:
        out = ISARA.Retr_PSD(
            [0.05, 0.10], [100.0, 50.0],
            dry_sca_coef=[5.0e-5],  # 5x the stub's scattering -> no CRI accepted
            dry_abs_coef=[1.0e-5],
            dry_wvl={"sca": [550], "abs": [520]},
            wet_sca_coef=[1.0e-5], wet_wvl={"sca": [550]},
            CRI_p=np.array([[1.50, 0.01]]),
            kappa_p=np.array([0.20]),
        )
    finally:
        ISARA.MMModel = saved
    assert out['attempt_flag_CRI_unitless'] == 1
    assert np.isnan(out["dry_RRI_unitless"])
    # kappa never attempted without a successful CRI
    assert out['attempt_flag_kappa_unitless'] == 0
    assert np.isnan(out["kappa_unitless"])


def test_retr_psd_validation_errors():
    err_cases = [
        # mismatched PSD lengths
        lambda: ISARA.Retr_PSD([0.05, 0.10], [1.0], [1e-5], [1e-5], {"sca": [550], "abs": [520]}),
        # duplicate radii
        lambda: ISARA.Retr_PSD([0.05, 0.05], [1.0, 2.0], [1e-5], [1e-5], {"sca": [550], "abs": [520]}),
        # fewer than 2 valid bins
        lambda: ISARA.Retr_PSD([0.05, np.nan], [1.0, 2.0], [1e-5], [1e-5], {"sca": [550], "abs": [520]}),
        # coefficient/channel length mismatch
        lambda: ISARA.Retr_PSD([0.05, 0.10], [1.0, 2.0], [1e-5, 1e-5], [1e-5], {"sca": [550], "abs": [520]}),
        # wet coefficients without wet_wvl
        lambda: ISARA.Retr_PSD([0.05, 0.10], [1.0, 2.0], [1e-5], [1e-5], {"sca": [550], "abs": [520]},
                               wet_sca_coef=[1e-5]),
    ]
    for case in err_cases:
        try:
            case()
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError from {case}")


def test_retr_psd_coarse_mode_note():
    calls = []
    saved = ISARA.MMModel
    ISARA.MMModel = _stub_model(calls)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ISARA.Retr_PSD(
                [0.05, 2.00], [100.0, 1.0],  # 2 um radius -> 4 um diameter (coarse)
                dry_sca_coef=[1.0e-5], dry_abs_coef=[1.0e-5],
                dry_wvl={"sca": [550], "abs": [520]},
                CRI_p=np.array([[1.50, 0.01]]),
            )
        assert "coarse" in buf.getvalue()
    finally:
        ISARA.MMModel = saved


def test_default_grids():
    cri = ISARA.default_CRI_grid()
    n_rri = len(np.unique(cri[:, 0]))
    n_iri = len(np.unique(cri[:, 1]))
    assert cri.shape == (n_rri*n_iri, 2)
    # note: this arange includes ~1.55 on most platforms (floating-point endpoint)
    assert np.isclose(cri[:, 0].min(), 1.51) and cri[:, 0].max() < 1.55 + 1e-9
    assert cri[:, 1].min() == 0 and np.isclose(cri[:, 1].max(), 0.030)
    kappa = ISARA.default_kappa_grid()
    assert kappa[0] == 0.0 and np.isclose(kappa[-1], 1.399) and len(kappa) == 1400


if __name__ == "__main__":
    test_fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in test_fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(test_fns) - failed}/{len(test_fns)} tests passed")
    sys.exit(1 if failed else 0)
