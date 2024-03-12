# Scripts and Instructions for Recreating Experiment and Analysis

## Reproduce my experiment
1. Install the software components required to conduct the experiment from [Contributing modeling software](../README.md#contributing-modeling-software)
2. Run the following script in the `workflow` directory to re-create the input configuration matrix (also see [details](#details-on-generating-input-configuration-matrix)):
   
| Script Name | Description | How to Run |
| --- | --- | --- |
| `make_perturbations.py` | generate the configuration matrix for the training ensemble | `python3 make_perturbations.py` |
3. Setup directories for each ensemble member and WRF `namelist.input` and FVCOM `run.nml`  with the physics options specified in the output from Step 2. Use initial/boundary conditions from [ERA5](http://doi.org/10.24381/cds.adbb2d47) for the time period 05/12/2018 00:00 UTC to 09/01/2018 00:00 UTC. Execute runs. 
4. Run the following Juptyer notebook scripts in the `workflow` directory to postprocess the WRF-FVCOM model output data:

| Script Name | Description | 
| --- | --- | 
| `process_WRF+FVCOM_global_daily_temperature_timeseries.ipynb` | Script to postprocess WRF+FVCOM ensemble outputs into daily surface temperature timeseries |



## Reproduce my analysis and figures
- Follow the steps to [reproduce the experiment](#reproduce-my-experiment) (Steps 1 & 3 are not trivial) --OR-- download the postprocessed output [data](../README.md#data-reference) from my experiment and place into `output` directory.
- Run the following Juptyer notebook scripts found in the `workflow` directory to reproduce the analysis and figures from the publication.

| Order | Script Name | Description | 
| --- | --- | --- |
| 1 | `plot_domain.ipynb` | Script to plot WRF and FVCOM computational domain |
| 2a | `daily_LST_surrogate_analysis.ipynb` | Script to generate the LST surrogate model |
| 2b | `daily_T2_surrogate_analysis.ipynb` | Script to generate the T2m surrogate model |
| 3 | `summary_global_sensitivity_analysis.ipynb` | Script to analyze the overall uncertainty/sensitivity |
| 4 | `spatial_variation_global_sensitivity_analysis.ipynb` | Script to analyze the spatial variation in uncertainty/sensitivity |
| 5 | `temporal_variation_global_sensitivity_analysis.ipynb` | Script to analyze the temporal variation in uncertainty/sensitivity |

## Details on generating input configuration matrix
Running `make_perturbations.py` will generate the perturbation matrix for all variables (there are 9) using a Korobov sequence with 18 samples which samples 89.5% of the range of each variable. Values for each perturbation are output into a netCDF file. This is the same idea as in Pringle et al. (2023)

One may view each variable and available options inside `wrf_fvcom/variables`. 

There are 5 different perturbations for WRF, each of these are treated as discrete uniform variables corresponding to the integer option in WRF namelist for the chosen parameterization scheme:
- WRF planetary bounday layer and surface layer scheme [discrete]
- WRF surface roughness (z0) scheme over water [discrete]
- WRF microphysics scheme [discrete]
- WRF radiation scheme [discrete]
- WRF land surface model [discrete]

There are 4 different perturbations for FVCOM, two of these are treated as discrete uniform variables corresponding to the compiler option for the chosen parameterization scheme and the other two are continous uniform variables corresponding to the float value used in the FVCOM namelist:
- FVCOM vertical mixing scheme [discrete]
- FVCOM bulk wind stress formulation [discrete]
- FVCOM shortwave radiation absorption: R [continuous]
- FVCOM Prandlt Number [continuous]

### Reference
1. Pringle, W. J., Burnett, Z., Sargsyan, K., Moghimi, S., & Myers, E. (2023). Efficient Probabilistic Prediction and Uncertainty Quantification of Tropical Cyclone-driven Storm Tides and Inundation. Artificial Intelligence for the Earth Systems, 2(2), e220040. https://doi.org/10.1175/AIES-D-22-0040.1
