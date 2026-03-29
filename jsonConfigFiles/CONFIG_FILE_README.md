# ISARA Configuration File

## Overview

The `RunISARA()` function now supports loading configuration from a JSON file, eliminating the need for tedious interactive input prompts.

## Usage

### With Configuration File

```python
import ISARA_Data_Retrieval

# Load configuration from file
ISARA_Data_Retrieval.RunISARA(config_file='my_config.json')
```

### Without Configuration File (Interactive Mode)

```python
import ISARA_Data_Retrieval

# Original interactive mode
ISARA_Data_Retrieval.RunISARA()
```

## Configuration File Format

The configuration file is a JSON file with the following structure:

```json
{
  "campaign_name": "ACTIVATE",
  "data_directory": "InsituData",
  "instruments": [
    {
      "name": "SMPS",
      "lower_bound_nm": 10,
      "upper_bound_nm": 500
    },
    {
      "name": "UHSAS",
      "lower_bound_nm": 500,
      "upper_bound_nm": 1000
    }
  ],
  "dry_channels": [
    {
      "scattering_wavelength_nm": 450,
      "absorption_wavelength_nm": 465,
      "color": "Blue"
    },
    {
      "scattering_wavelength_nm": 550,
      "absorption_wavelength_nm": 520,
      "color": "Green"
    },
    {
      "scattering_wavelength_nm": 700,
      "absorption_wavelength_nm": 660,
      "color": "Red"
    }
  ],
  "wet_channels": [
    {
      "scattering_wavelength_nm": 450,
      "color": "Blue"
    }
  ],
  "additional_wavelengths": {
    "wavelengths_nm": [370, 530, 1060],
    "colors": ["Blue", "Green", "Red"]
  }
}
```

## Field Descriptions

### Required Fields

- **`campaign_name`** (string): Name of the campaign (e.g., "ACTIVATE")
- **`data_directory`** (string): Name of the directory containing ICARTT files (e.g., "InsituData")
- **`instruments`** (array): List of size distribution instruments
  - **`name`** (string): Instrument name (must match CSV file in `./misc/{campaign}/SDBinInfo/`)
  - **`lower_bound_nm`** (number): Lower size cutoff in nanometers
  - **`upper_bound_nm`** (number): Upper size cutoff in nanometers
- **`dry_channels`** (array): List of dry spectral channels
  - **`scattering_wavelength_nm`** (integer): Scattering wavelength in nanometers
  - **`absorption_wavelength_nm`** (integer): Absorption wavelength in nanometers
  - **`color`** (string): Color name (e.g., "Blue", "Green", "Red")
- **`wet_channels`** (array): List of humidified spectral channels
  - **`scattering_wavelength_nm`** (integer): Scattering wavelength in nanometers
  - **`color`** (string): Color name

### Optional Fields

- **`additional_wavelengths`** (object or null): Additional wavelengths for validation
  - **`wavelengths_nm`** (array of integers): List of wavelengths in nanometers
  - **`colors`** (array of strings): List of color names (must match length of wavelengths_nm)
  - If not needed, set to `null` or omit the field

## Example Configuration Files

A template file `isara_config_template.json` is provided in the repository. You can copy and modify it for your specific campaign.

## Notes

1. **Size Bin Files**: The instrument names in the config file must match the CSV files in `./misc/{campaign_name}/SDBinInfo/`. For example, if you specify `"name": "SMPS"`, there must be a file like `SMPS_bins.csv` in that directory.

2. **Size Cutoffs**: The `lower_bound_nm` and `upper_bound_nm` values should be chosen to avoid overlaps between instruments (to prevent double-counting) and gaps (to prevent underestimation). See `CODE_REVIEW_ISSUES.md` for details on this important consideration.

3. **Wavelength Colors**: The color names are used in the output metadata for netCDF compliance. Common values are "Blue", "Green", "Red", but any string is acceptable.

4. **Backward Compatibility**: If `config_file=None` (the default), the function behaves exactly as before, prompting for all inputs interactively.

## Error Handling

- If the configuration file is not found, a `FileNotFoundError` is raised.
- If the JSON file is malformed, a `json.JSONDecodeError` is raised.
- Missing required fields will cause errors when the code tries to access them.

