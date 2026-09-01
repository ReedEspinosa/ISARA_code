import mopsmap_wrapper
MMModel = mopsmap_wrapper.Model
import numpy as np

def default_CRI_grid(rri_min=1.51, rri_max=1.54, rri_step=0.01):
  """
  Returns the default grid of candidate complex refractive index (CRI) values searched by Retr_CRI.
  The RRI range is parameterizable (defaults preserve the historical 1.51-1.54 grid); the IRI
  grid is fixed at 0, 1e-7..1e-4 decades, then 0.001-0.030 in 0.001 steps.

  :return: 2-D array of shape (N, 2) where column 0 is RRI and column 1 is IRI
  :rtype: numpy array
  """
  RRIp = np.arange(rri_min, rri_max + rri_step/2.0, rri_step).reshape(-1)
  IRIp = np.hstack((0, 10**(-7), 10**(-6), 10**(-5), 10**(-4), np.arange(0.001, 0.031, 0.001).reshape(-1)))
  CRI_p = np.empty((len(IRIp)*len(RRIp), 2))
  io = 0
  for i1 in range(len(IRIp)):
    for i2 in range(len(RRIp)):
      CRI_p[io, :] = [RRIp[i2], IRIp[i1]]
      io += 1
  return CRI_p

def default_kappa_grid():
  """
  Returns the default grid of candidate hygroscopicity (kappa) values searched by Retr_kappa.

  :return: 1-D array of kappa values
  :rtype: numpy array
  """
  return np.arange(0.0, 1.40, 0.001).reshape(-1)

def _output_wavelengths(wvl_dry, wvl_wet, val_wvl, out_wvl):
  """Sorted union (nm, int) of every wavelength the retrieval should report."""
  wl = list(wvl_dry["sca"]) + list(wvl_dry["abs"])
  if wvl_wet is not None:
    wl += list(wvl_wet["sca"])
  if val_wvl is not None:
    wl += list(np.asarray(val_wvl).astype(int))
  if out_wvl is not None:
    wl += list(np.asarray(out_wvl).astype(int))
  return sorted(set(int(x) for x in wl))

def Retr_PSD(radii_um,
  dndlogdp_cm3,
  dry_sca_coef,
  dry_abs_coef,
  dry_wvl,
  wet_sca_coef=None,
  wet_wvl=None,
  RH_wet=80,
  RH_ambient=None,
  CRI_p=None,
  kappa_p=None,
  val_wvl=None,
  out_wvl=None,
  size_equ='cs',
  nonabs_fraction=0,
  shape='sphere',
  rho_dry=1.0,
  rho_wet=1.0,
  num_theta=2,
  path_optical_dataset='./optical_dataset/',
  path_mopsmap_executable='./mopsmap',
  lut=None,
  forward_engine='mopsmap',
  estimator='linf-mean',
  sca_sigma=None,
  abs_sigma=None,
  wet_sigma=None,
  obs_cov=None,
):

  """
  Instrument-agnostic, single-time-point ISARA retrieval from a directly supplied particle size
  distribution (PSD). Retrieves the dry complex refractive index (CRI) and, if a humidified
  scattering coefficient is supplied, the hygroscopicity parameter kappa. The PSD is passed to
  MOPSMAP as a single mode, so dN/dlogDp is interpolated between the supplied bin locations and
  integrated from the smallest to the largest bin.

  :param radii_um: Bin center locations as RADII in micrometer (converted to diameters internally)
  :type radii_um: numpy array
  :param dndlogdp_cm3: dN/dlogDp number concentrations in cm^-3 at each bin (NaN bins are dropped)
  :type dndlogdp_cm3: numpy array
  :param dry_sca_coef: Dry scattering coefficients in m^-1, one per dry_wvl["sca"] channel
  :type dry_sca_coef: numpy array
  :param dry_abs_coef: Dry absorption coefficients in m^-1, one per dry_wvl["abs"] channel
  :type dry_abs_coef: numpy array
  :param dry_wvl: Dictionary with keys "sca" and "abs" giving the channel wavelengths in nm
  :type dry_wvl: dict
  :param wet_sca_coef: Optional humidified scattering coefficients in m^-1, one per wet_wvl["sca"] channel; if provided (and the CRI retrieval succeeds) a kappa retrieval is attempted
  :type wet_sca_coef: numpy array
  :param wet_wvl: Dictionary with key "sca" giving the humidified channel wavelengths in nm; required when wet_sca_coef is provided
  :type wet_wvl: dict
  :param RH_wet: Percent relative humidity of the humidified scattering measurement (default 80)
  :type RH_wet: double
  :param RH_ambient: Optional ambient percent relative humidity; when finite (and < 100) and the
    kappa retrieval succeeds, the humidified state is also forward-calculated at this RH and
    reported under 'amb_cal_*' keys (plus 'amb_gf/RRI/IRI_unitless' and 'RH_ambient_percent')
  :type RH_ambient: double
  :param out_wvl: Optional array of wavelengths (nm) at which the full dry/wet/ambient optical
    set is reported, in addition to the measured and validation wavelengths
  :type out_wvl: numpy array
  :param CRI_p: Optional 2-D array of candidate (RRI, IRI) pairs; defaults to default_CRI_grid()
  :type CRI_p: numpy array
  :param kappa_p: Optional 1-D array of candidate kappa values; defaults to default_kappa_grid()
  :type kappa_p: numpy array
  :param val_wvl: Optional array of additional output wavelengths in nm
  :type val_wvl: numpy array
  :param size_equ: Size equivalence (e.g., 'cs')
  :type size_equ: str
  :param nonabs_fraction: Non-absorbing fraction of the PSD
  :type nonabs_fraction: int
  :param shape: Particle shape string passed to MOPSMAP (e.g., 'sphere')
  :type shape: str
  :param rho_dry: Dry particle density in g cm^-3 (only affects MOPSMAP mass output, not the optics)
  :type rho_dry: double
  :param rho_wet: Humidified particle density in g cm^-3
  :type rho_wet: double
  :param num_theta: Number of phase function angles to provide
  :type num_theta: int
  :param path_optical_dataset: Path of the optical dataset required for MOPSMAP
  :type path_optical_dataset: str
  :param path_mopsmap_executable: Path of the mopsmap executable
  :type path_mopsmap_executable: str
  :param lut: Optional precomputed optics_lut.OpticsLUT for the CRI grid search. Used only when its fingerprint (bin diameters after NaN dropping, wavelengths, CRI grid, shape assumptions) matches this call; otherwise the per-candidate MOPSMAP path runs as usual, so passing a LUT never changes which retrievals are possible
  :type lut: optics_lut.OpticsLUT
  :param forward_engine: 'mopsmap' (default; Fortran subprocess per forward call) or 'table'
    (in-process NumPy integration over the extracted single-sphere efficiency table in
    mopsmap_sphere_table/ -- ~100x faster, spheres only, validated to <=0.21% vs exact Mie).
    All retrieval logic (grids, acceptance criteria, estimator) is engine-independent.
  :type forward_engine: str
  :param estimator: Solution selection for both the CRI grid search and the kappa scan.
    'linf-mean' (default, historical): CRI = mean of candidates with ALL channels inside
    tolerance (20% sca, 1 Mm-1 abs; re-verified at the mean); kappa = first grid value with
    all humidified channels within 1%. 'chi2-wmean': Gaussian-posterior weighted mean over
    the grid with sigma equal to those same tolerances, success gate min reduced chi^2 <= 1;
    adds dry_CRI_min_chi2 / kappa_min_chi2 / kappa_std outputs, and the *_accepted_std
    outputs become posterior-weighted stds. Best RMSE and ~+30% more successful retrievals
    in the ASCENT-ACP estimator study (scripts/estimator_study.py, 2026-08-31); evaluates
    the full kappa grid, so pair it with forward_engine='table'.
  :type estimator: str
  :param sca_sigma: Optional per-channel 1-sigma uncertainties (m^-1, order of dry_wvl['sca'])
    used by the 'chi2-wmean' estimator in place of the default 20% relative tolerance;
    abs_sigma likewise replaces the 1 Mm-1 absolute tolerance and wet_sigma (order of
    wet_wvl['sca']) the 1% humidified tolerance. None keeps the legacy tolerances.
  :type sca_sigma: numpy array
  :param obs_cov: Optional full (2*n_ch x 2*n_ch) observation+model covariance in (m^-1)^2,
    channel order [sca channels..., abs channels...]. When given, the chi2-wmean CRI search
    uses the generalized chi^2 r' S^-1 r / n_ch, which MARGINALIZES over correlated model
    uncertainties (e.g. the LAS diameter-scale and impactor-parameter terms) encoded as
    rank-1 outer products in S: residual patterns matching a known nuisance direction are
    forgiven, inconsistent spectral shapes still fail. Overrides sca_sigma/abs_sigma in the
    CRI stage (the kappa stage keeps wet_sigma).
  :type obs_cov: numpy array
  :return: Dictionary with retrieved dry CRI (and kappa when attempted), calculated coefficients/SSA at the measured (and validation) wavelengths, and attempt flags (0 no attempt, 1 attempt, 2 success); failed values are NaN
  :rtype: dict

  NOTES on coarse-mode PSDs: the PSD is NOT restricted to the fine mode, but the caller must
  ensure physical consistency with the optical measurements:
  1) The optical instruments must have actually sampled the supplied size range (inlet cutpoint);
     coarse bins the optics never saw bias the retrieved CRI low in RRI.
  2) Nephelometer angular truncation (~7-170 deg) undercounts the forward-scattering peak of
     super-micrometer particles, a low bias in measured scattering that grows with coarse fraction.
  3) With substantial coarse fraction, the single wavelength-independent CRI, the spherical shape,
     and the single bulk kappa (one growth factor applied to ALL bins in Retr_kappa) become
     scattering-weighted effective values rather than intrinsic properties.
  """

  ## prepare and validate the PSD; sizes are kept in micrometer throughout
  radii = np.asarray(radii_um, dtype=float).reshape(-1)
  dndlogdp = np.asarray(dndlogdp_cm3, dtype=float).reshape(-1)
  if radii.size != dndlogdp.size:
    raise ValueError(f"radii_um (n={radii.size}) and dndlogdp_cm3 (n={dndlogdp.size}) must have the same length.")
  valid = np.isfinite(radii) & (radii > 0) & np.isfinite(dndlogdp)
  radii = radii[valid]
  dndlogdp = dndlogdp[valid]
  if radii.size < 2:
    raise ValueError("At least 2 valid (finite, positive radius) PSD bins are required.")
  order = np.argsort(radii)
  radii = radii[order]
  dndlogdp = dndlogdp[order]
  if np.any(np.diff(radii) <= 0):
    raise ValueError("radii_um contains duplicate values; bin locations must be unique.")
  sd = {"PSD": np.multiply(dndlogdp, pow(10, 6))} # cm^-3 -> m^-3
  dpg = {"PSD": np.multiply(radii, 2.0)} # radius -> diameter, micrometer
  if dpg["PSD"][-1] > 2.5:
    print(f'NOTE: PSD extends to {dpg["PSD"][-1]:.2f} um diameter (coarse mode). Ensure the optical '
          'instruments sampled these sizes (inlet cutpoint) and note that nephelometer truncation, '
          'the spherical shape assumption, and the single bulk CRI/kappa weaken for coarse particles.')

  ## collect channel wavelengths and measured coefficients
  wvl_dry = {"sca": np.asarray(dry_wvl["sca"]).astype(int), "abs": np.asarray(dry_wvl["abs"]).astype(int)}
  dry_sca = np.asarray(dry_sca_coef, dtype=float).reshape(-1)
  dry_abs = np.asarray(dry_abs_coef, dtype=float).reshape(-1)
  if (dry_sca.size != wvl_dry["sca"].size) or (dry_abs.size != wvl_dry["abs"].size):
    raise ValueError("dry_sca_coef/dry_abs_coef lengths must match dry_wvl['sca']/dry_wvl['abs'].")
  optical_measurements = {}
  for i2 in range(wvl_dry["sca"].size):
    optical_measurements[f'dry_meas_sca_coef_{wvl_dry["sca"][i2]}_m-1'] = dry_sca[i2]
  for i2 in range(wvl_dry["abs"].size):
    optical_measurements[f'dry_meas_abs_coef_{wvl_dry["abs"][i2]}_m-1'] = dry_abs[i2]

  ## validate the optional humidified inputs up front, before any MOPSMAP work
  wvl_wet = None
  wet_sca = None
  if wet_sca_coef is not None:
    if wet_wvl is None:
      raise ValueError("wet_wvl must be provided when wet_sca_coef is given.")
    wvl_wet = {"sca": np.asarray(wet_wvl["sca"]).astype(int)}
    wet_sca = np.asarray(wet_sca_coef, dtype=float).reshape(-1)
    if wet_sca.size != wvl_wet["sca"].size:
      raise ValueError("wet_sca_coef length must match wet_wvl['sca'].")

  if CRI_p is None:
    CRI_p = default_CRI_grid()
  if kappa_p is None:
    kappa_p = default_kappa_grid()
  if val_wvl is not None:
    val_wvl = np.asarray(val_wvl).astype(int)

  ## per-mode parameter dictionaries for the single "PSD" mode
  Size_equ = {"PSD": size_equ}
  Nonabs_fraction = {"PSD": nonabs_fraction}
  Shape = {"PSD": shape}
  Rho_dry = {"PSD": rho_dry}
  Rho_wet = {"PSD": rho_wet}

  ## resolve the forward model callable once; every downstream stage uses it
  if forward_engine == 'table':
    import sphere_optics
    model = sphere_optics.Model
  elif forward_engine == 'mopsmap':
    model = MMModel
  else:
    raise ValueError(f"forward_engine must be 'mopsmap' or 'table', got '{forward_engine}'")

  finalout = {}
  finalout['attempt_flag_CRI_unitless'] = 1
  finalout['attempt_flag_kappa_unitless'] = 0
  Results = Retr_CRI(wvl_dry, val_wvl, optical_measurements, sd, dpg, CRI_p, Size_equ,
    Nonabs_fraction, Shape, Rho_dry, num_theta, path_optical_dataset, path_mopsmap_executable,
    lut=lut, model=model, estimator=estimator,
    sca_sigma=sca_sigma, abs_sigma=abs_sigma, obs_cov=obs_cov)
  for key in Results:
    finalout[key] = Results[key]
  cri_success = Results["dry_RRI_unitless"] is not None
  if cri_success:
    finalout['attempt_flag_CRI_unitless'] = 2
    ## complete the dry optical set: derive sca/abs from ext and SSA at every
    ## wavelength where the CRI search reported them (sca channels lack abs
    ## and vice versa); setdefault never overwrites a directly stored value
    for w in _output_wavelengths(wvl_dry, wvl_wet, val_wvl, out_wvl):
      ext = finalout.get(f'dry_cal_ext_coef_{w}_m-1')
      ssa = finalout.get(f'dry_cal_SSA_{w}_unitless')
      if ext is not None and ssa is not None:
        finalout.setdefault(f'dry_cal_sca_coef_{w}_m-1', ssa*ext)
        finalout.setdefault(f'dry_cal_abs_coef_{w}_m-1', (1.0-ssa)*ext)

  ## kappa retrieval if humidified scattering is supplied and the dry CRI was retrieved
  if wet_sca is not None:
    for i2 in range(wvl_wet["sca"].size):
      optical_measurements[f'wet_meas_sca_coef_{wvl_wet["sca"][i2]}_m-1'] = wet_sca[i2]
      finalout[f'wet_meas_sca_coef_{wvl_wet["sca"][i2]}_m-1'] = wet_sca[i2]
    finalout["kappa_unitless"] = None
    for i2 in range(wvl_wet["sca"].size):
      finalout[f'wet_cal_sca_coef_{wvl_wet["sca"][i2]}_m-1'] = None
      finalout[f'wet_cal_SSA_{wvl_wet["sca"][i2]}_unitless'] = None
      finalout[f'wet_cal_ext_coef_{wvl_wet["sca"][i2]}_m-1'] = None
    if cri_success and np.all(np.isfinite(wet_sca)) and np.all(wet_sca > 0):
      finalout['attempt_flag_kappa_unitless'] = 1
      CRI_dry = np.array([Results["dry_RRI_unitless"], Results["dry_IRI_unitless"]])
      KResults = Retr_kappa(wvl_wet, val_wvl, optical_measurements, sd, dpg, RH_wet, kappa_p,
        CRI_dry, Size_equ, Nonabs_fraction, Shape, Rho_wet, num_theta,
        path_optical_dataset, path_mopsmap_executable, model=model,
        estimator=estimator, wet_sigma=wet_sigma)
      for key in KResults:
        finalout[key] = KResults[key]
      if KResults["kappa_unitless"] is not None:
        finalout['attempt_flag_kappa_unitless'] = 2
        ## report the full humidified state (all output wavelengths, humidified
        ## CRI, growth factor) at the retrieval RH, and optionally at ambient RH
        kappa_ret = KResults["kappa_unitless"]
        all_wvl = _output_wavelengths(wvl_dry, wvl_wet, val_wvl, out_wvl)
        states = [("wet", RH_wet)]
        if RH_ambient is not None and np.isfinite(RH_ambient) and 0 < RH_ambient < 100:
          states.append(("amb", RH_ambient))
          finalout['RH_ambient_percent'] = float(RH_ambient)
        for tag, rh_state in states:
          hum = humidified_optics(sd, dpg, CRI_dry, kappa_ret, rh_state, all_wvl,
            Size_equ, Nonabs_fraction, Shape, Rho_wet, num_theta,
            path_optical_dataset, path_mopsmap_executable, model=model)
          for gkey in ("gf_unitless", "RRI_unitless", "IRI_unitless"):
            finalout[f'{tag}_{gkey}'] = hum[gkey]
          for key, value in hum.items():
            if key.startswith(("sca_coef_", "abs_coef_", "ext_coef_", "SSA_")):
              finalout[f'{tag}_cal_{key}'] = value

  ## failed/unattempted retrievals are reported as NaN
  for key in finalout:
    if finalout[key] is None:
      finalout[key] = np.nan
  return finalout

def Retr_CRI(wvl_dict,
  val_wvl, 
  optical_measurements,
  sd,
  dpg,
  CRI_p,
  size_equ,
  nonabs_fraction,
  shape,
  rho,
  num_theta,
  path_optical_dataset,
  path_mopsmap_executable,
  lut=None,
  model=None,
  estimator='linf-mean',
  sca_sigma=None,
  abs_sigma=None,
  obs_cov=None,
):

  """
  Returns aerosol particle real and imaginary refractive index from three scattering coefficeint measurements, three absorption coefficient measurements, a measured number concentration for an aerosol size distribution. WARNINGS: 1) numpy must be installed to the python environment 2) mopsmap_wrapper.py must be present in a directory that is in your PATH
  
  :param wvl_dict: Dictionary of wavelengths associated with each of the scattering and absorption measurements
  :type wvl_dict: numpy dictionary
  :param val_wvl: Dictionary of wavelengths associated with validation measurements
  :type val_wvl: numpy dictionary
  :param optical_measurements: Dictionary containing measured dry scattering and absorption coefficients in m^-1; NOTE: There should be one key per channel (e.g., optical_measurements['dry_meas_sca_coef_450_m-1'], optical_measurements['dry_meas_abs_coef_470_m-1'], etc.)
  :type optical_measurements: numpy dictionary       
  :param sd: Dictionary containing the modal size resolved number concentrations in m^-3; NOTE: there should be one key for each measurement mode
  :type sd: numpy dictionary  
  :param dpg: Dictionary containing the modal geometric mean particle diameters of each size bin in micrometer; NOTE: there should be one key for each measurement mode
  :type dpg: numpy dictionary
  :param CRI_p: 2-D array containing the prescribed RRI and IRI range to be searched
  :type CRI_p: numpy array
  :param nonabs_fraction: Dictionary of integers indicating the desired non-absorbing fraction for each size mode; NOTE: there should be one key for each measurement mode
  :type nonabs_fraction: numpy dictionary
  :param shape: String indicating the desired particle shape(s) for each size mode; NOTE: there should be one key for each measurement mode
  :type shape: numpy dictionary
  :param rho: Double indicating the desired particle density in g cm^-3 for each size mode; NOTE: there should be one key for each measurement mode
  :type rho: numpy dictionary 
  :param num_theta: Integer indicating the number of phase function angles to provide
  :type num_theta: numpy int     
  :param path_optical_dataset: String indicating the path for the optical dataset required for MOPSMAP
  :type path_optical_dataset: str
  :param path_mopsmap_executable: String indicating the path for the mopsmap.exe file
  :type path_mopsmap_executable: str                                  
  :return: Dictionary (Results) with the retrieved complex refractive index, calculated scattering and absorption coefficients in native measurements, and calculated single scattering albedo and extinction coefficients in measured and validation wavelengths
  :rtype: numpy dictionary
  """

  if model is None:
    model = MMModel
  L1 = len(CRI_p[:,0]) # length of array with all possible cri values
  L2 = len(wvl_dict["sca"]) # number of scattering channels
  L2a = len(wvl_dict["abs"]) # number of absorption channels (may differ)
  wvl = np.unique(np.concatenate([np.asarray(wvl_dict["sca"]),
                                  np.asarray(wvl_dict["abs"])]))
  ## Prepare output arrays and dictionary
  iri = np.full((L1), np.nan)
  rri = np.full((L1), np.nan)
  Results = dict()  
  Results["dry_RRI_unitless"] = None
  Results["dry_IRI_unitless"] = None
  ref_scat_coef = np.full((L2),np.nan)## measured scattering coefficients
  ref_abs_coef = np.full((L2a),np.nan) ## measured absorption coefficients
  for i2 in range(L2):
    Results[f'dry_cal_sca_coef_{wvl_dict["sca"][i2]}_m-1'] = None
    Results[f'dry_cal_SSA_{wvl_dict["sca"][i2]}_unitless'] = None
    Results[f'dry_cal_ext_coef_{wvl_dict["sca"][i2]}_m-1'] = None
    ref_scat_coef[i2] = optical_measurements[f'dry_meas_sca_coef_{wvl_dict["sca"][i2]}_m-1']
  for i2 in range(L2a):
    Results[f'dry_cal_abs_coef_{wvl_dict["abs"][i2]}_m-1'] = None
    Results[f'dry_cal_SSA_{wvl_dict["abs"][i2]}_unitless'] = None
    Results[f'dry_cal_ext_coef_{wvl_dict["abs"][i2]}_m-1'] = None
    ref_abs_coef[i2] = optical_measurements[f'dry_meas_abs_coef_{wvl_dict["abs"][i2]}_m-1']
  ## Decide whether the (optional) precomputed optics LUT can replace the
  ## per-candidate MOPSMAP subprocess calls of the grid search. The LUT is
  ## applicable only for a single-mode PSD on exactly its bin grid with the
  ## same wavelengths, CRI grid and particle assumptions; in every other case
  ## the original subprocess loop below runs unchanged.
  use_lut = False
  if lut is not None and len(sd) == 1:
    lut_mode = next(iter(sd))
    use_lut = (
      np.all(np.isfinite(sd[lut_mode]))
      and lut.matches(wvl, CRI_p, dpg[lut_mode], size_equ[lut_mode],
                      nonabs_fraction[lut_mode], shape[lut_mode])
    )

  scat_coef_all = np.full((L1, L2), np.nan)
  abs_coef_all = np.full((L1, L2a), np.nan)
  if use_lut:
    ## vectorized grid search: coefficients for ALL candidates as dot products
    ext_all, sca_all = lut.coefficients(sd[lut_mode])  # (L1, n_wvl) each
    i_sca = [lut.wavelength_index(wvl_dict["sca"][i2]) for i2 in range(L2)]
    i_abs = [lut.wavelength_index(wvl_dict["abs"][i2]) for i2 in range(L2a)]
    scat_coef_all = sca_all[:, i_sca]
    abs_coef_all = ext_all[:, i_abs] - sca_all[:, i_abs]
  else:
   for i1 in range(L1): # loop through possible cri values; store per-candidate coefficients
    RRI_p = {}
    IRI_p = {}
    for imode in sd:
      RRI_p[imode] = CRI_p[i1,0]
      IRI_p[imode] = CRI_p[i1,1]
    results = model(wvl,size_equ,sd,dpg,RRI_p,IRI_p,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable)
    for i2 in range(L2):
      scat_coef_all[i1, i2] = results[f'ssa_{wvl_dict["sca"][i2]}']*results[f'ext_coeff_{wvl_dict["sca"][i2]}_m-1']
    for i2 in range(L2a):
      abs_coef_all[i1, i2] = results[f'ext_coeff_{wvl_dict["abs"][i2]}_m-1']-results[f'ssa_{wvl_dict["abs"][i2]}']*results[f'ext_coeff_{wvl_dict["abs"][i2]}_m-1']

  ## shared misfit measures for both estimators (identical tolerances:
  ## 20% relative per sca channel, 1 Mm-1 absolute per abs channel)
  Cdif1 = np.divide(abs(ref_scat_coef - scat_coef_all), ref_scat_coef,
    out=np.full_like(scat_coef_all, np.inf), where=ref_scat_coef > 1e-10)
  Cdif2 = abs(ref_abs_coef - abs_coef_all)

  rri = None
  iri = None
  if estimator == 'linf-mean':
    ## historical ISARA selection: accept candidates with ALL channels inside
    ## tolerance, report their unweighted mean (re-verified below)
    valid = np.logical_and((Cdif1 < 0.2).all(axis=1), (Cdif2 < pow(10, -6)).all(axis=1))
    if np.any(valid):
      Results["dry_CRI_n_accepted_unitless"] = int(np.sum(valid))
      Results["dry_RRI_accepted_std_unitless"] = float(np.std(CRI_p[valid, 0]))
      Results["dry_IRI_accepted_std_unitless"] = float(np.std(CRI_p[valid, 1]))
      rri = np.mean(CRI_p[valid, 0])
      iri = np.mean(CRI_p[valid, 1])
  elif estimator == 'chi2-wmean':
    ## Gaussian-posterior mean on the grid: reduced chi^2 with sigma equal to
    ## the historical tolerances; success gate min reduced chi^2 <= 1; weights
    ## exp(-n_ch*chi2/2). Continuous weights remove the acceptance-boundary
    ## fragility of the binary gate and lower RMSE (scripts/estimator_study.py
    ## in ASCENT-ACP, 2026-08-31).
    n_ch = L2 + L2a
    if obs_cov is not None:
      ## generalized chi^2 with the full observation+model covariance:
      ## residual vector r = [sca..., abs...] per candidate
      r = np.hstack([scat_coef_all - ref_scat_coef.reshape(1, L2),
                     abs_coef_all - ref_abs_coef.reshape(1, L2a)])
      S_inv = np.linalg.inv(np.asarray(obs_cov, float))
      chi2 = np.einsum('ki,ij,kj->k', r, S_inv, r) / n_ch
    elif sca_sigma is not None:
      ## instrument-model sigmas (absolute, m^-1); Cdif1 is |dy|/ref so undo
      sig_s = np.asarray(sca_sigma, float).reshape(1, L2)
      chi2 = (((Cdif1 * ref_scat_coef) / sig_s) ** 2).sum(axis=1)
      if abs_sigma is not None:
        sig_a = np.asarray(abs_sigma, float).reshape(1, L2a)
        chi2 = chi2 + ((Cdif2 / sig_a) ** 2).sum(axis=1)
      else:
        chi2 = chi2 + ((Cdif2 / pow(10, -6)) ** 2).sum(axis=1)
      chi2 /= n_ch
    else:
      chi2 = ((Cdif1 / 0.2) ** 2).sum(axis=1)
      if abs_sigma is not None:
        sig_a = np.asarray(abs_sigma, float).reshape(1, L2a)
        chi2 = chi2 + ((Cdif2 / sig_a) ** 2).sum(axis=1)
      else:
        chi2 = chi2 + ((Cdif2 / pow(10, -6)) ** 2).sum(axis=1)
      chi2 /= n_ch
    k_best = int(np.nanargmin(chi2))
    Results["dry_CRI_min_chi2_unitless"] = float(chi2[k_best])
    if chi2[k_best] <= 1.0:
      w = np.exp(-0.5 * n_ch * (chi2 - chi2[k_best]))
      w = w / w.sum()
      rri = float(w @ CRI_p[:, 0])
      iri = float(w @ CRI_p[:, 1])
      Results["dry_CRI_n_accepted_unitless"] = int(np.sum(chi2 <= 1.0))
      Results["dry_RRI_accepted_std_unitless"] = float(np.sqrt(w @ (CRI_p[:, 0] - rri) ** 2))
      Results["dry_IRI_accepted_std_unitless"] = float(np.sqrt(w @ (CRI_p[:, 1] - iri) ** 2))
  else:
    raise ValueError(f"estimator must be 'linf-mean' or 'chi2-wmean', got '{estimator}'")

  if rri is not None: # a solution was selected; forward-calculate at the reported CRI
    RRI_d = {}
    IRI_d = {}
    for imode in sd:
      RRI_d[imode] = rri
      IRI_d[imode] = iri
    results = model(wvl,size_equ,sd,dpg,RRI_d,IRI_d,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable)
    scat_coef = np.full((L2),np.nan)
    abs_coef = np.full((L2a),np.nan)
    for i2 in range(L2):
      scat_coef[i2] = results[f'ssa_{wvl_dict["sca"][i2]}']*results[f'ext_coeff_{wvl_dict["sca"][i2]}_m-1']
    for i2 in range(L2a):
      abs_coef[i2] = results[f'ext_coeff_{wvl_dict["abs"][i2]}_m-1']-results[f'ssa_{wvl_dict["abs"][i2]}']*results[f'ext_coeff_{wvl_dict["abs"][i2]}_m-1']

    ## the linf estimator historically re-verifies that the mean CRI itself
    ## passes the acceptance test (the accepted set can be non-convex); the
    ## chi2 estimator's gate is min chi^2 and needs no re-verification
    accept = True
    if estimator == 'linf-mean':
      Cd1 = np.divide(abs(ref_scat_coef-scat_coef), ref_scat_coef, out=np.full_like(ref_scat_coef, np.inf), where=ref_scat_coef>1e-10)
      Cd2 = abs(ref_abs_coef-abs_coef)
      accept = bool((Cd1 < 0.2).all() and (Cd2 < pow(10, -6)).all())
    if accept: # store dry cri and dry calculated coefficients and SSA in all measured wavelengths
      Results["dry_RRI_unitless"] = rri
      Results["dry_IRI_unitless"] = iri
      for i2 in range(L2):
        Results[f'dry_cal_sca_coef_{wvl_dict["sca"][i2]}_m-1'] = results[f'ssa_{wvl_dict["sca"][i2]}']*results[f'ext_coeff_{wvl_dict["sca"][i2]}_m-1']
        Results[f'dry_cal_SSA_{wvl_dict["sca"][i2]}_unitless'] = results[f'ssa_{wvl_dict["sca"][i2]}']
        Results[f'dry_cal_ext_coef_{wvl_dict["sca"][i2]}_m-1'] = results[f'ext_coeff_{wvl_dict["sca"][i2]}_m-1']
      for i2 in range(L2a):
        Results[f'dry_cal_abs_coef_{wvl_dict["abs"][i2]}_m-1'] = results[f'ext_coeff_{wvl_dict["abs"][i2]}_m-1']-results[f'ssa_{wvl_dict["abs"][i2]}']*results[f'ext_coeff_{wvl_dict["abs"][i2]}_m-1']
        Results[f'dry_cal_SSA_{wvl_dict["abs"][i2]}_unitless'] = results[f'ssa_{wvl_dict["abs"][i2]}']
        Results[f'dry_cal_ext_coef_{wvl_dict["abs"][i2]}_m-1'] = results[f'ext_coeff_{wvl_dict["abs"][i2]}_m-1']
      if val_wvl is not None: # if validation wavelengths are requested, provide outputs for those wavelengths as well
        results = model(val_wvl,size_equ,sd,dpg,RRI_d,IRI_d,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable)
        for iwvl in range(len(val_wvl)):
          Results[f'dry_cal_sca_coef_{val_wvl[iwvl]}_m-1'] = results[f'ssa_{val_wvl[iwvl]}']*results[f'ext_coeff_{val_wvl[iwvl]}_m-1']
          Results[f'dry_cal_SSA_{val_wvl[iwvl]}_unitless'] = results[f'ssa_{val_wvl[iwvl]}']
          Results[f'dry_cal_ext_coef_{val_wvl[iwvl]}_m-1'] = results[f'ext_coeff_{val_wvl[iwvl]}_m-1']

  return Results # return dictionary (Results) of dry cri and dry calculated extinction, scattering, and absorption coefficients and SSA in all measured and validation wavelengths


def Retr_kappa(wvl_dict,
  val_wvl, 
  optical_measurements,
  sd,
  dpg,
  RH,
  kappa_p,
  CRI_d,
  size_equ,
  nonabs_fraction,
  shape,
  rho,
  num_theta,
  path_optical_dataset, 
  path_mopsmap_executable,
  model=None,
  estimator='linf-mean',
  wet_sigma=None,
):
  """
  Returns aerosol particle hygroscopic growth factor from a humdified scattering coefficeint measurement, dry complex refractive index, and a measured number concentration for an aerosol size distribution. WARNINGS: 1) numpy must be installed to the python environment 2) mopsmap_wrapper.py must be present in a directory that is in your PATH.
  
  :param wvl_dict: Dictionary of wavelengths associated with each of the scattering and absorption measurements
  :type wvl_dict: numpy dictionary
  :param val_wvl: Dictionary of wavelengths associated with validation measurements
  :type val_wvl: numpy dictionary
  :param optical_measurements: Dictionary containing measured dry scattering and absorption coefficients in m^-1; NOTE: There should be one key per channel (e.g., optical_measurements['wet_meas_sca_coef_450_m-1'] etc.)
  :type optical_measurements: numpy dictionary       
  :param sd: Dictionary containing the modal size resolved number concentrations in m^-3; NOTE: there should be one key for each measurement mode
  :type sd: numpy dictionary  
  :param dpg: Dictionary containing the modal geometric mean particle diameters of each size bin in micrometer; NOTE: there should be one key for each measurement mode
  :type dpg: numpy dictionary
  :param RH: Array containing the percent relative humidity associated with the measured humidified scattering coefficients
  :type RH: int   
  :param kappa_p: Array containing the desired kappa range to be searched.
  :type kappa_p: numpy array    
  :param CRI_d: Array containing the desired dry RRI and IRI.
  :type CRI_d: numpy array
  :param nonabs_fraction: Dictionary of integers indicating the desired non-absorbing fraction for each size mode; NOTE: there should be one key for each measurement mode
  :type nonabs_fraction: numpy dictionary
  :param shape: String indicating the desired particle shape(s) for each size mode; NOTE: there should be one key for each measurement mode
  :type shape: numpy dictionary
  :param rho: Double indicating the desired particle density in g cm^-3 for each size mode; NOTE: there should be one key for each measurement mode
  :type rho: numpy dictionary 
  :param num_theta: Integer indicating the number of phase function angles to provide
  :type num_theta: int       
  :param path_optical_dataset: String indicating the path for the optical dataset required for MOPSMAP
  :type path_optical_dataset: str
  :param path_mopsmap_executable: String indicating the path for the mopsmap.exe file
  :type path_mopsmap_executable: str                                         
  :return: Real refractive index, imaginary refractive index, calculated scattering and absorption coefficients in native measurements, and calculated single scattering albedo and extinction coefficients in all wavelengths
  :rtype: numpy dictionary
  """

  if model is None:
    model = MMModel
  L1 = len(kappa_p) # length of array with all possible kappa values
  L2 = len(wvl_dict["sca"]) # number of measured scattering (Sc) coefficient channels

  ## collect scattering coefficient channel wavelengths (wvl) into array and sort in ascending order
  wvl = None
  for i1 in range(L2):
    if i1 == 0:
      wvl = np.array([wvl_dict["sca"][i1]])
    else:
      wvl = np.hstack((wvl,np.array([wvl_dict["sca"][i1]])))
  wvl = np.sort(wvl, axis=None)
  ##
  ## Prepare output dictionary
  Results = dict()
  Results["kappa_unitless"] = None
  for i2 in range(L2):
    Results[f'wet_cal_sca_coef_{wvl_dict["sca"][i2]}_m-1'] = None
    Results[f'wet_cal_SSA_{wvl_dict["sca"][i2]}_unitless'] = None
    Results[f'wet_cal_ext_coef_{wvl_dict["sca"][i2]}_m-1'] = None
  ##  
  RRIw = 1.33 # set rri of water
  IRIw = 0 # set iri of water

  def _wet_state(kappa_val):
    """gf-grown diameters and water-volume-mixed CRI for one kappa."""
    gf = np.power((1+kappa_val*RH/(100-RH)),1/3)
    dpg_w = {}
    RRI_w = {}
    IRI_w = {}
    for imode in sd:
      dpg_w[imode] = np.squeeze(np.multiply(gf,dpg[imode]))
      RRI_w[imode] = (CRI_d[0]+((gf**3)-1)*RRIw)/(gf**3)
      IRI_w[imode] = (CRI_d[1]+((gf**3)-1)*IRIw)/(gf**3)
    return dpg_w, RRI_w, IRI_w

  ref_scat_coef = np.array([optical_measurements[f'wet_meas_sca_coef_{wvl_dict["sca"][i2]}_m-1']
                            for i2 in range(L2)], dtype=float)

  def _wet_sca(kappa_val):
    dpg_w, RRI_w, IRI_w = _wet_state(kappa_val)
    results = model(wvl,size_equ,sd,dpg_w,RRI_w,IRI_w,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable)
    return np.array([results[f'ssa_{wvl_dict["sca"][i2]}']*results[f'ext_coeff_{wvl_dict["sca"][i2]}_m-1']
                     for i2 in range(L2)])

  kappa_sel = None
  if estimator == 'linf-mean':
    ## historical selection: ascending scan, FIRST kappa with all humidified
    ## scattering channels within 1% of the measurement
    for i1 in range(L1):
      scat_coef = _wet_sca(kappa_p[i1])
      Cdif = np.divide(abs(ref_scat_coef-scat_coef), ref_scat_coef, out=np.full_like(ref_scat_coef, np.inf), where=ref_scat_coef>1e-10)
      if np.all(Cdif<0.01):
        kappa_sel = kappa_p[i1]
        break
  elif estimator == 'chi2-wmean':
    ## Gaussian-posterior mean over the kappa grid, sigma = the historical 1%
    ## per-channel tolerance; success gate min reduced chi^2 <= 1. Evaluates
    ## the full grid (use the 'table' forward engine; per-candidate subprocess
    ## calls make this path slow under the 'mopsmap' engine).
    chi2 = np.full(L1, np.inf)
    if wet_sigma is not None:
      sig_w = np.asarray(wet_sigma, float).reshape(-1)
    for i1 in range(L1):
      scat_coef = _wet_sca(kappa_p[i1])
      if wet_sigma is not None:
        chi2[i1] = np.mean(((scat_coef - ref_scat_coef) / sig_w) ** 2)
      else:
        Cdif = np.divide(abs(ref_scat_coef-scat_coef), ref_scat_coef, out=np.full_like(ref_scat_coef, np.inf), where=ref_scat_coef>1e-10)
        chi2[i1] = np.mean((Cdif/0.01)**2)
    k_best = int(np.nanargmin(chi2))
    Results["kappa_min_chi2_unitless"] = float(chi2[k_best])
    if chi2[k_best] <= 1.0:
      w = np.exp(-0.5*L2*(chi2 - chi2[k_best]))
      w = w / w.sum()
      kappa_sel = float(w @ kappa_p)
      Results["kappa_std_unitless"] = float(np.sqrt(w @ (kappa_p - kappa_sel)**2))
  else:
    raise ValueError(f"estimator must be 'linf-mean' or 'chi2-wmean', got '{estimator}'")

  if kappa_sel is not None:
    Results["kappa_unitless"] = kappa_sel
    ## forward-calculate the reported humidified state at the selected kappa
    dpg_w, RRI_w, IRI_w = _wet_state(kappa_sel)
    results = model(wvl,size_equ,sd,dpg_w,RRI_w,IRI_w,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable)
    for i2 in range(L2):
      Results[f'wet_cal_sca_coef_{wvl_dict["sca"][i2]}_m-1'] = results[f'ssa_{wvl_dict["sca"][i2]}']*results[f'ext_coeff_{wvl_dict["sca"][i2]}_m-1']
      Results[f'wet_cal_SSA_{wvl_dict["sca"][i2]}_unitless'] = results[f'ssa_{wvl_dict["sca"][i2]}']
      Results[f'wet_cal_ext_coef_{wvl_dict["sca"][i2]}_m-1'] = results[f'ext_coeff_{wvl_dict["sca"][i2]}_m-1']
    if val_wvl is not None:
      results = model(val_wvl,size_equ,sd,dpg_w,RRI_w,IRI_w,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable)
      for iwvl in range(len(val_wvl)):
        Results[f'wet_cal_sca_coef_{val_wvl[iwvl]}_m-1'] = results[f'ssa_{val_wvl[iwvl]}']*results[f'ext_coeff_{val_wvl[iwvl]}_m-1']
        Results[f'wet_cal_SSA_{val_wvl[iwvl]}_unitless'] = results[f'ssa_{val_wvl[iwvl]}']
        Results[f'wet_cal_ext_coef_{val_wvl[iwvl]}_m-1'] = results[f'ext_coeff_{val_wvl[iwvl]}_m-1']

  return Results # return dictionary (Results) of kappa and wet calculated extinction, scattering, and absorption coefficients and SSA in all measured and validation wavelengths

def humidified_optics(sd,
  dpg,
  CRI_d,
  kappa,
  RH,
  wvl,
  size_equ,
  nonabs_fraction,
  shape,
  rho,
  num_theta,
  path_optical_dataset,
  path_mopsmap_executable,
  model=None,
):
  """
  Forward-calculate humidified optical properties at an arbitrary relative humidity.

  Grows the dry PSD by the kappa-Kohler growth factor gf = (1 + kappa*RH/(100-RH))^(1/3),
  mixes the dry CRI with water (RRI 1.33, IRI 0) by volume, and runs MOPSMAP at the
  requested wavelengths. This is the same humidification model used inside Retr_kappa,
  exposed so callers can compute the humidified state at any RH (e.g. ambient) and any
  wavelength set from an already-retrieved dry CRI and kappa.

  :param sd: Dictionary of modal size resolved number concentrations in m^-3
  :param dpg: Dictionary of DRY modal geometric mean bin diameters in micrometer
  :param CRI_d: Array [RRI_dry, IRI_dry]
  :param kappa: Retrieved (or assumed) hygroscopicity parameter
  :param RH: Percent relative humidity of the humidified state (must be < 100)
  :param wvl: Array of output wavelengths in nm
  :return: Dictionary with 'gf_unitless', 'RRI_unitless', 'IRI_unitless' (humidified CRI)
    and per-wavelength 'sca_coef_{w}_m-1', 'abs_coef_{w}_m-1', 'ext_coef_{w}_m-1',
    'SSA_{w}_unitless'
  :rtype: dict
  """
  if model is None:
    model = MMModel
  wvl = np.unique(np.asarray(wvl).astype(int))
  gf = np.power((1 + kappa * RH / (100 - RH)), 1/3)
  RRIw = 1.33 # rri of water
  IRIw = 0.0 # iri of water
  dpg_w = {}
  RRI_w = {}
  IRI_w = {}
  for imode in sd:
    dpg_w[imode] = np.squeeze(np.multiply(gf, dpg[imode]))
    RRI_w[imode] = (CRI_d[0]+((gf**3)-1)*RRIw)/(gf**3) # volume weighted humidified rri
    IRI_w[imode] = (CRI_d[1]+((gf**3)-1)*IRIw)/(gf**3) # volume weighted humidified iri
  results = model(wvl,size_equ,sd,dpg_w,RRI_w,IRI_w,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable)
  first_mode = next(iter(sd))
  out = {
    "gf_unitless": float(gf),
    "RRI_unitless": float(RRI_w[first_mode]),
    "IRI_unitless": float(IRI_w[first_mode]),
  }
  for w in wvl:
    ext = results[f'ext_coeff_{w}_m-1']
    ssa = results[f'ssa_{w}']
    out[f'sca_coef_{w}_m-1'] = ssa*ext
    out[f'abs_coef_{w}_m-1'] = (1.0-ssa)*ext
    out[f'ext_coef_{w}_m-1'] = ext
    out[f'SSA_{w}_unitless'] = ssa
  return out
