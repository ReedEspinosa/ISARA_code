"""Precomputed single-scattering lookup table (LUT) for the ISARA CRI search.

MOPSMAP integrates the supplied dN/dlogDp with linear interpolation between
bin centers, so the integrated extinction/scattering coefficients are exactly
linear in the per-bin concentrations. This module exploits that: for a fixed
(wavelengths, CRI candidate grid, bin diameters, shape, ...) configuration it
runs MOPSMAP once per (candidate, unit "hat" bin) and stores the resulting
kernels. Afterwards the coefficients for ANY size distribution on that bin
grid are dot products -- no Fortran subprocess in the per-sample search loop.

The LUT is strictly optional: ISARA.Retr_PSD/Retr_CRI accept ``lut=None`` (the
default) and then behave exactly as before, calling MOPSMAP per candidate.
A supplied LUT is used only when its fingerprint matches the current call
(same bin diameters after NaN dropping, same wavelengths, same CRI grid and
particle assumptions); otherwise the code silently falls back to subprocess
calls, so partially-missing PSDs remain fully supported.

Numerical fidelity: superposition reproduces direct MOPSMAP runs to ~1e-5
relative (limited only by the 4-significant-digit formatting of the MOPSMAP
input files), far below the 20% / 1 Mm^-1 retrieval match tolerances.
"""

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import mopsmap_wrapper

_HAT_M3 = 1.0e6  # build hats of 1 cm^-3 (in m^-3); kernels normalized back out

# module-level state for worker processes
_W = {}


def _fingerprint(wvl_nm, cri_grid, dpg_um, size_equ, nonabs_fraction, shape,
                 rho, num_theta):
    blob = json.dumps(
        {
            "wvl_nm": np.asarray(wvl_nm, float).round(6).tolist(),
            "cri_grid": np.asarray(cri_grid, float).round(10).tolist(),
            "dpg_um": np.asarray(dpg_um, float).round(10).tolist(),
            "size_equ": str(size_equ),
            "nonabs_fraction": float(nonabs_fraction),
            "shape": str(shape),
            "rho": float(rho),
            "num_theta": int(num_theta),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class OpticsLUT:
    """Kernels K_ext/K_sca of shape (n_candidates, n_wavelengths, n_bins).

    ``coefficients(sd_m3)`` returns (ext, sca) arrays of shape
    (n_candidates, n_wavelengths) in m^-1 for a dN/dlogDp vector in m^-3
    given on exactly ``dpg_um``.
    """

    def __init__(self, wvl_nm, cri_grid, dpg_um, K_ext, K_sca, size_equ="cs",
                 nonabs_fraction=0, shape="sphere", rho=1.0, num_theta=2):
        self.wvl_nm = np.asarray(wvl_nm, float)
        self.cri_grid = np.asarray(cri_grid, float)
        self.dpg_um = np.asarray(dpg_um, float)
        self.K_ext = np.asarray(K_ext, float)
        self.K_sca = np.asarray(K_sca, float)
        self.size_equ = str(size_equ)
        self.nonabs_fraction = float(nonabs_fraction)
        self.shape = str(shape)
        self.rho = float(rho)
        self.num_theta = int(num_theta)
        expected = (len(self.cri_grid), len(self.wvl_nm), len(self.dpg_um))
        if self.K_ext.shape != expected or self.K_sca.shape != expected:
            raise ValueError(
                f"Kernel shape {self.K_ext.shape} != expected {expected}"
            )

    @property
    def fingerprint(self):
        return _fingerprint(self.wvl_nm, self.cri_grid, self.dpg_um,
                            self.size_equ, self.nonabs_fraction, self.shape,
                            self.rho, self.num_theta)

    def matches(self, wvl_nm, cri_grid, dpg_um, size_equ, nonabs_fraction,
                shape):
        """True when this LUT is applicable to a Retr_CRI call."""
        wvl_nm = np.asarray(wvl_nm, float)
        cri_grid = np.asarray(cri_grid, float)
        dpg_um = np.asarray(dpg_um, float)
        return (
            wvl_nm.shape == self.wvl_nm.shape
            and np.allclose(wvl_nm, self.wvl_nm)
            and cri_grid.shape == self.cri_grid.shape
            and np.allclose(cri_grid, self.cri_grid)
            and dpg_um.shape == self.dpg_um.shape
            and np.allclose(dpg_um, self.dpg_um, rtol=1e-9)
            and str(size_equ) == self.size_equ
            and float(nonabs_fraction) == self.nonabs_fraction
            and str(shape) == self.shape
        )

    def coefficients(self, sd_m3):
        """(ext, sca) in m^-1 for all candidates; sd_m3 is dN/dlogDp in m^-3."""
        sd = np.asarray(sd_m3, float)
        if sd.shape != self.dpg_um.shape:
            raise ValueError(f"sd shape {sd.shape} != bins {self.dpg_um.shape}")
        if not np.all(np.isfinite(sd)):
            raise ValueError("sd contains non-finite values; use the fallback path")
        return self.K_ext @ sd, self.K_sca @ sd

    def wavelength_index(self, wvl):
        idx = np.nonzero(np.isclose(self.wvl_nm, float(wvl)))[0]
        if idx.size != 1:
            raise KeyError(f"wavelength {wvl} not in LUT ({self.wvl_nm})")
        return int(idx[0])

    def save(self, path):
        np.savez_compressed(
            path,
            wvl_nm=self.wvl_nm,
            cri_grid=self.cri_grid,
            dpg_um=self.dpg_um,
            K_ext=self.K_ext,
            K_sca=self.K_sca,
            meta=json.dumps(
                {
                    "size_equ": self.size_equ,
                    "nonabs_fraction": self.nonabs_fraction,
                    "shape": self.shape,
                    "rho": self.rho,
                    "num_theta": self.num_theta,
                    "fingerprint": self.fingerprint,
                }
            ),
        )
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            return cls(
                z["wvl_nm"], z["cri_grid"], z["dpg_um"], z["K_ext"], z["K_sca"],
                size_equ=meta["size_equ"],
                nonabs_fraction=meta["nonabs_fraction"],
                shape=meta["shape"],
                rho=meta["rho"],
                num_theta=meta["num_theta"],
            )


def _worker_init(seed_offset, scratch_dir):
    if scratch_dir:
        os.makedirs(scratch_dir, exist_ok=True)
        os.chdir(scratch_dir)
    # mopsmap_wrapper names temp files from np.random; make workers unique
    np.random.seed((os.getpid() + seed_offset) & 0xFFFFFFFF)


def _run_one_candidate(args):
    """All hat-bin MOPSMAP runs for one CRI candidate; returns kernel slices."""
    (i_cand, rri, iri, wvl_nm, dpg_um, size_equ, nonabs_fraction, shape, rho,
     num_theta, path_optical_dataset, path_mopsmap_executable) = args
    n_wvl, n_bin = len(wvl_nm), len(dpg_um)
    k_ext = np.zeros((n_wvl, n_bin))
    k_sca = np.zeros((n_wvl, n_bin))
    for i_bin in range(n_bin):
        hat = np.zeros(n_bin)
        hat[i_bin] = _HAT_M3
        res = mopsmap_wrapper.Model(
            np.asarray(wvl_nm, int), {"PSD": size_equ}, {"PSD": hat},
            {"PSD": np.asarray(dpg_um, float)}, {"PSD": rri}, {"PSD": iri},
            {"PSD": nonabs_fraction}, {"PSD": shape}, {"PSD": rho}, 0, 0,
            num_theta, path_optical_dataset, path_mopsmap_executable,
        )
        for i_wvl, w in enumerate(np.asarray(wvl_nm, int)):
            ext = res[f"ext_coeff_{w}_m-1"]
            ssa = res[f"ssa_{w}"]
            k_ext[i_wvl, i_bin] = ext / _HAT_M3
            k_sca[i_wvl, i_bin] = ssa * ext / _HAT_M3
    return i_cand, k_ext, k_sca


def build(wvl_nm, cri_grid, dpg_um, path_optical_dataset,
          path_mopsmap_executable, size_equ="cs", nonabs_fraction=0,
          shape="sphere", rho=1.0, num_theta=2, n_workers=1, scratch_dir=None,
          cache_dir=None, verbose=True):
    """Build (or load from ``cache_dir``) the LUT for one configuration.

    The cache filename is derived from a fingerprint of all inputs, so any
    change in bins/wavelengths/grid/assumptions produces a fresh build.
    """
    fp = _fingerprint(wvl_nm, cri_grid, dpg_um, size_equ, nonabs_fraction,
                      shape, rho, num_theta)
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"optics_lut_{fp}.npz")
        if os.path.exists(cache_path):
            if verbose:
                print(f"OpticsLUT: loading cached {cache_path}")
            return OpticsLUT.load(cache_path)

    cri_grid = np.asarray(cri_grid, float)
    n_cand = len(cri_grid)
    if verbose:
        print(
            f"OpticsLUT: building {n_cand} candidates x {len(dpg_um)} bins "
            f"x {len(wvl_nm)} wavelengths ({n_cand * len(dpg_um)} MOPSMAP runs)"
        )
    jobs = [
        (i, cri_grid[i, 0], cri_grid[i, 1], list(map(float, wvl_nm)),
         list(map(float, dpg_um)), size_equ, nonabs_fraction, shape, rho,
         num_theta, path_optical_dataset, path_mopsmap_executable)
        for i in range(n_cand)
    ]
    K_ext = np.zeros((n_cand, len(wvl_nm), len(dpg_um)))
    K_sca = np.zeros_like(K_ext)
    if n_workers <= 1:
        prev = os.getcwd()
        if scratch_dir:
            os.makedirs(scratch_dir, exist_ok=True)
            os.chdir(scratch_dir)
        try:
            for job in jobs:
                i, k_ext, k_sca = _run_one_candidate(job)
                K_ext[i], K_sca[i] = k_ext, k_sca
        finally:
            os.chdir(prev)
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(12345, scratch_dir),
        ) as pool:
            for i, k_ext, k_sca in pool.map(_run_one_candidate, jobs, chunksize=1):
                K_ext[i], K_sca[i] = k_ext, k_sca

    lut = OpticsLUT(wvl_nm, cri_grid, dpg_um, K_ext, K_sca, size_equ=size_equ,
                    nonabs_fraction=nonabs_fraction, shape=shape, rho=rho,
                    num_theta=num_theta)
    if cache_path:
        lut.save(cache_path)
        if verbose:
            print(f"OpticsLUT: cached to {cache_path}")
    return lut
