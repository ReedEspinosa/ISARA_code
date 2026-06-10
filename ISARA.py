import mopsmap_wrapper
MMModel = mopsmap_wrapper.Model
import numpy as np

def default_CRI_grid():
  """
  Returns the default grid of candidate complex refractive index (CRI) values searched by Retr_CRI.

  :return: 2-D array of shape (N, 2) where column 0 is RRI and column 1 is IRI
  :rtype: numpy array
  """
  RRIp = np.arange(1.51, 1.55, 0.01).reshape(-1)
  IRIp = np.hstack((0, 10**(-7), 10**(-6), 10**(-5), 10**(-4), np.arange(0.001, 0.081, 0.001).reshape(-1)))
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

def Retr_PSD(radii_um,
  dndlogdp_cm3,
  dry_sca_coef,
  dry_abs_coef,
  dry_wvl,
  wet_sca_coef=None,
  wet_wvl=None,
  RH_wet=80,
  CRI_p=None,
  kappa_p=None,
  val_wvl=None,
  size_equ='cs',
  nonabs_fraction=0,
  shape='sphere',
  rho_dry=1.0,
  rho_wet=1.0,
  num_theta=2,
  path_optical_dataset='./optical_dataset/',
  path_mopsmap_executable='./mopsmap',
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

  finalout = {}
  finalout['attempt_flag_CRI_unitless'] = 1
  finalout['attempt_flag_kappa_unitless'] = 0
  Results = Retr_CRI(wvl_dry, val_wvl, optical_measurements, sd, dpg, CRI_p, Size_equ,
    Nonabs_fraction, Shape, Rho_dry, num_theta, path_optical_dataset, path_mopsmap_executable)
  for key in Results:
    finalout[key] = Results[key]
  cri_success = Results["dry_RRI_unitless"] is not None
  if cri_success:
    finalout['attempt_flag_CRI_unitless'] = 2

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
        path_optical_dataset, path_mopsmap_executable)
      for key in KResults:
        finalout[key] = KResults[key]
      if KResults["kappa_unitless"] is not None:
        finalout['attempt_flag_kappa_unitless'] = 2

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

  L1 = len(CRI_p[:,0]) # length of array with all possible cri values
  L2 = len(wvl_dict["sca"]) # number of measured scattering (Sc) coefficient channels
  ## collect scattering and absorption (Abs) coefficient channel wavelengths (wvl) into array
  wvl = None
  for iwvl in range(L2):
    if iwvl == 0:
      wvl = np.array([wvl_dict["sca"][iwvl],wvl_dict["abs"][iwvl]])
    else:
      wvl = np.hstack((wvl,np.array([wvl_dict["sca"][iwvl],wvl_dict["abs"][iwvl]])))
  ##
  wvl = np.sort(wvl, axis=None) # sort array of wavelengths in ascending order
  ## Prepare output arrays and dictionary
  iri = np.full((L1), np.nan)
  rri = np.full((L1), np.nan)
  Results = dict()  
  Results["dry_RRI_unitless"] = None
  Results["dry_IRI_unitless"] = None
  ref_scat_coef = np.full((L2),np.nan)## prepare arrays of measured scattering and absorption coefficients
  ref_abs_coef = np.full((L2),np.nan)  
  for i2 in range(L2):
    Results[f'dry_cal_sca_coef_{wvl_dict["sca"][i2]}_m-1'] = None
    Results[f'dry_cal_abs_coef_{wvl_dict["abs"][i2]}_m-1'] = None
    Results[f'dry_cal_SSA_{wvl_dict["sca"][i2]}_unitless'] = None
    Results[f'dry_cal_SSA_{wvl_dict["abs"][i2]}_unitless'] = None
    Results[f'dry_cal_ext_coef_{wvl_dict["sca"][i2]}_m-1'] = None
    Results[f'dry_cal_ext_coef_{wvl_dict["abs"][i2]}_m-1'] = None
    ## Assign values to prepared measured coefficients
    ref_scat_coef[i2] = optical_measurements[f'dry_meas_sca_coef_{wvl_dict["sca"][i2]}_m-1']
    ref_abs_coef[i2] = optical_measurements[f'dry_meas_abs_coef_{wvl_dict["abs"][i2]}_m-1']
    ##
  ##
  for i1 in range(L1): # initiate loop through possible cri values
    ## assign the rri and iri for to each size mode.
    RRI_p = {}
    IRI_p = {}
    for imode in sd:
      #if imode == 'SMPS':
        #RRI_p[imode] = 1.4
        #IRI_p[imode] = 0.01       
      #else:
      RRI_p[imode] = CRI_p[i1,0]
      IRI_p[imode] = CRI_p[i1,1]
    ##  
    results = MMModel(wvl,size_equ,sd,dpg,RRI_p,IRI_p,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable) # calculate microphysical properties for a given cri
    scat_coef = np.full((L2),np.nan)## prepare arrays of calculated scattering and absorption coefficients
    abs_coef = np.full((L2),np.nan)  
    ## Assign values to prepared calculated coefficients
    for i2 in range(L2):
      scat_coef[i2] = results[f'ssa_{wvl_dict["sca"][i2]}']*results[f'ext_coeff_{wvl_dict["sca"][i2]}_m-1']
      abs_coef[i2] = results[f'ext_coeff_{wvl_dict["abs"][i2]}_m-1']-results[f'ssa_{wvl_dict["abs"][i2]}']*results[f'ext_coeff_{wvl_dict["abs"][i2]}_m-1'] 
      ref_scat_coef[i2] = optical_measurements[f'dry_meas_sca_coef_{wvl_dict["sca"][i2]}_m-1']
      ref_abs_coef[i2] = optical_measurements[f'dry_meas_abs_coef_{wvl_dict["abs"][i2]}_m-1']

    ##
    Cdif1 = np.divide(abs(ref_scat_coef-scat_coef), ref_scat_coef, out=np.full_like(ref_scat_coef, np.inf), where=ref_scat_coef>1e-10) # calculate absolute relative difference of scattering coefficients in each channel
    Cdif2 = abs(ref_abs_coef-abs_coef)# calculate absolute difference of absoprtion coefficients in each channel

    ## check if relative difference in scattering coefficient is within 20% for all and channels that the difference in absorption coefficient is within 1 Mm-1 for all channels
    a1 = ((Cdif1)<0.2).astype('int')#a1[np.isinf(a1)]=0
    a2 = ((Cdif2)<pow(10,-6)).astype('int')#
    if (np.sum(a1) == L2) and (np.sum(a2) == L2):
      iri[i1] = CRI_p[i1,1]
      rri[i1] = CRI_p[i1,0]
    ##    

  flgs = np.logical_not(np.isnan(rri)) # flag valid solutions
  
  #pause()
  #print(ref_abs_coef*10**6,abs_coef*10**6,'\n')
  if np.sum(rri[flgs])>0: # check to see if any valid solutions exist 
    ## take mean rri and iri of all valid solutions and recalculate aerosol properties with mean cri values.    
    rri = np.mean(rri[flgs]) 
    iri = np.mean(iri[flgs])
    RRI_d = {}
    IRI_d = {}
    for imode in sd:
      RRI_d[imode] = rri
      IRI_d[imode] = iri
    results = MMModel(wvl,size_equ,sd,dpg,RRI_d,IRI_d,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable) 
    ##
    ## same as before, check for to ensure recalculated scattering coefficients are within 20% and absorption coefficients are with 1 Mm-1 when using mean cri
    scat_coef = np.full((L2),np.nan)
    abs_coef = np.full((L2),np.nan)

    for i2 in range(L2):
      scat_coef[i2] = results[f'ssa_{wvl_dict["sca"][i2]}']*results[f'ext_coeff_{wvl_dict["sca"][i2]}_m-1']
      abs_coef[i2] = results[f'ext_coeff_{wvl_dict["abs"][i2]}_m-1']-results[f'ssa_{wvl_dict["abs"][i2]}']*results[f'ext_coeff_{wvl_dict["abs"][i2]}_m-1'] 

    Cd1 = np.divide(abs(ref_scat_coef-scat_coef), ref_scat_coef, out=np.full_like(ref_scat_coef, np.inf), where=ref_scat_coef>1e-10)
    Cd2 = abs(ref_abs_coef-abs_coef)
    a1 = ((Cd1)<0.2).astype('int')
    #a1[np.isinf(a1)]=0
    a2 = ((Cd2)<pow(10,-6)).astype('int')
    if (np.sum(a1) == L2) and (np.sum(a2) == L2): # if solution is valid, store dry cri and dry calculated extinction, scattering, and absorption coefficients and SSA in all measured wavelengths
      Results["dry_RRI_unitless"] = rri
      Results["dry_IRI_unitless"] = iri
      for i2 in range(L2):
        Results[f'dry_cal_sca_coef_{wvl_dict["sca"][i2]}_m-1'] = results[f'ssa_{wvl_dict["sca"][i2]}']*results[f'ext_coeff_{wvl_dict["sca"][i2]}_m-1']
        Results[f'dry_cal_abs_coef_{wvl_dict["abs"][i2]}_m-1'] = results[f'ext_coeff_{wvl_dict["abs"][i2]}_m-1']-results[f'ssa_{wvl_dict["abs"][i2]}']*results[f'ext_coeff_{wvl_dict["abs"][i2]}_m-1'] 
        Results[f'dry_cal_SSA_{wvl_dict["sca"][i2]}_unitless'] = results[f'ssa_{wvl_dict["sca"][i2]}']
        Results[f'dry_cal_SSA_{wvl_dict["abs"][i2]}_unitless'] = results[f'ssa_{wvl_dict["abs"][i2]}']
        Results[f'dry_cal_ext_coef_{wvl_dict["sca"][i2]}_m-1'] = results[f'ext_coeff_{wvl_dict["sca"][i2]}_m-1']
        Results[f'dry_cal_ext_coef_{wvl_dict["abs"][i2]}_m-1'] = results[f'ext_coeff_{wvl_dict["abs"][i2]}_m-1']
      if val_wvl is not None: # if validation wavelengths are requested, provide outputs for those wavelengths as well
        wvl2 = None
        for iwvl in range(len(val_wvl)):
          if iwvl == 0:
            wvl2 = val_wvl
          else:
            wvl2 = np.hstack((wvl2,val_wvl))
        results = MMModel(wvl2,size_equ,sd,dpg,RRI_d,IRI_d,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable) 
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
  stop_indx = 0 # initate stop index for first valid solution
  RRIw = 1.33 # set rri of water 
  IRIw = 0 # set iri of water 
  for i1 in range(L1): # loop through each possible kappa value 
    dpg_w = {}
    RRI_w = {}
    IRI_w = {}
    for imode in sd:
      gf = np.power((1+kappa_p[i1]*RH/(100-RH)),1/3) # calculate growth factor given the incrimental kappa value and the measurement relative humidity for each size mode
      #if imode == 'SMPS':
        #dpg_w[imode] = dpg[imode] # adjust the size distribution by multplying the growth factor by each dry particle diameter in each size mode
        #RRI_w[imode] = CRI_d[0] # volume weighted humidified rri for each size mode
        #IRI_w[imode] = CRI_d[1] # volume weighted humidified iri for each size mode
      #else:
      dpg_w[imode] = np.squeeze(np.multiply(gf,dpg[imode])) # adjust the size distribution by multplying the growth factor by each dry particle diameter in each size mode
      RRI_w[imode] = (CRI_d[0]+((gf**3)-1)*RRIw)/(gf**3) # volume weighted humidified rri for each size mode
      IRI_w[imode] = (CRI_d[1]+((gf**3)-1)*IRIw)/(gf**3) # volume weighted humidified iri for each size mode
    if stop_indx == 0: # stop if last solution was valid (Cdif<0.01)
      results = MMModel(wvl,size_equ,sd,dpg_w,RRI_w,IRI_w,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable) # calculate microphysical properties for a given kappa
      scat_coef = np.full((L2),np.nan)# prepare array of calculated scattering coefficients
      ref_scat_coef = np.full((L2),np.nan) # prepare array of measured scattering coefficients
      ## Assign values to prepared measured and calculated coefficients
      for i2 in range(L2):
        scat_coef[i2] = results[f'ssa_{wvl_dict["sca"][i2]}']*results[f'ext_coeff_{wvl_dict["sca"][i2]}_m-1']
        ref_scat_coef[i2] = optical_measurements[f'wet_meas_sca_coef_{wvl_dict["sca"][i2]}_m-1']
      ##  
      Cdif = np.divide(abs(ref_scat_coef-scat_coef), ref_scat_coef, out=np.full_like(ref_scat_coef, np.inf), where=ref_scat_coef>1e-10) # calculate absolute relative difference of scattering coefficients in each channel
      if np.all(Cdif<0.01): # solution is valid if scattering coefficients are within 1%
        Results["kappa_unitless"] = kappa_p[i1] # store retrieved kappa
        ## store calculated scattering and extinction coefficients and SSA for measured and validation wavelengths
        for i2 in range(L2):
          Results[f'wet_cal_sca_coef_{wvl_dict["sca"][i2]}_m-1'] = results[f'ssa_{wvl_dict["sca"][i2]}']*results[f'ext_coeff_{wvl_dict["sca"][i2]}_m-1'] 
          Results[f'wet_cal_SSA_{wvl_dict["sca"][i2]}_unitless'] = results[f'ssa_{wvl_dict["sca"][i2]}']
          Results[f'wet_cal_ext_coef_{wvl_dict["sca"][i2]}_m-1'] = results[f'ext_coeff_{wvl_dict["sca"][i2]}_m-1']
        if val_wvl is not None:
          wvl2 = None
          for iwvl in range(len(val_wvl)):
            if iwvl == 0:
              wvl2 = val_wvl
            else:
              wvl2 = np.hstack((wvl2,val_wvl))
          results = MMModel(wvl2,size_equ,sd,dpg_w,RRI_w,IRI_w,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable) 
          for iwvl in range(len(val_wvl)):
            Results[f'wet_cal_sca_coef_{val_wvl[iwvl]}_m-1'] = results[f'ssa_{val_wvl[iwvl]}']*results[f'ext_coeff_{val_wvl[iwvl]}_m-1']
            Results[f'wet_cal_SSA_{val_wvl[iwvl]}_unitless'] = results[f'ssa_{val_wvl[iwvl]}']
            Results[f'wet_cal_ext_coef_{val_wvl[iwvl]}_m-1'] = results[f'ext_coeff_{val_wvl[iwvl]}_m-1'] 
        ##        
        stop_indx = 1 # change stop index when first valid solution is reached
  return Results # return dictionary (Results) of kappa and wet calculated extinction, scattering, and absorption coefficients and SSA in all measured and validation wavelengths
