[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10806950.svg)](https://doi.org/10.5281/zenodo.10806950)

# Pringle-etal_2024_JAMES

<p align="center">
  <a href="https://compass.pnnl.gov/GLM/COMPASSGLM"><img alt="GreatLakesUQ" src="figures/overall_sensitivities_w_interactions.png" width="100%"></a>
</p>

**Coupled Lake-Atmosphere-Land Physics Uncertainties in a Great Lakes Regional Climate Model**

William Pringle<sup>1\*</sup>, [COMPASS-GLM Team 1](https://compass.pnnl.gov/GLM/Team)<sup>1,2,3</sup>, and Khachik Sargsyan<sup>4</sup>

<sup>1 </sup> Environmental Science Division, Argonne National Laboratory, Lemont, IL, USA
<sup>2 </sup> Department of Civil, Environmental and Geospatial Engineering, Michigan Technological University, Houghton, MI, USA
<sup>3 </sup> Environmental Science Division, Argonne National Laboratory, Lemont, IL, USA
<sup>4 </sup> Pacific Northwest National Laboratory, Richland, WA, USA

\* corresponding author:  wpringle@anl.gov

## Abstract
This study develops a surrogate-based method to assess the uncertainty within a convective permitting integrated modeling system of the Great Lakes region, arising from interacting physics parameterizations across the lake, atmosphere, and land surface. Perturbed physics ensembles of the model during the 2018 summer are used to train a neural network surrogate model to predict lake surface temperature (LST) and near-surface air temperature (T2m). Average physics uncertainties are determined to be 1.5&deg;C for LST and T2m over land, and 1.9&deg;C for T2m over lake, but these have significant spatiotemporal variations. We find that atmospheric physics parameterizations are the dominant sources of uncertainty for both LST and T2m, and there is a substantial atmosphere-lake physics interaction component. LST and T2m over the lake is more uncertain in the deeper northern lakes particularly during the rapid warming phase that occurs in late spring/early summer. The LST uncertainty increases with sensitivity to the lake model's surface wind stress scheme. T2m over land is more uncertain over forested areas in the north, where it is most sensitive to the land surface model, than the more agricultural land in the south, where it is most sensitive to the atmospheric planetary boundary and surface layer scheme. Uncertainty also increases in the southwest during multiday temperature declines with higher sensitivity to the land surface model. Last, we show that the deduced physics uncertainty of T2m is statistically smaller than a regional warming perturbation exceeding 0.5&deg;C.

## Journal reference
Pringle, W. J., Huang, C., Xue, P., Wang, J., Sargsyan K., Kayastha, M., Chakraborty, T. C., Yang, Z., Qian Y., Hetland, R. D. (2024). Coupled Lake-Atmosphere-Land Physics Uncertainties in a Great Lakes Regional Climate Model. Submitted to Journal of Advances in Modeling Earth Systems, ESSOAr DOI to add.

## Code reference
Pringle W. J. (2024, March 11). COMPASS-DOE/GreatLakes_CoupledModel_Uncertainty (Version v0.1). [Zenodo](https://doi.org/10.5281/zenodo.10806950).

## Data reference
Pringle W. J. (2024, March 11). Great Lakes WRF-FVCOM model ensemble outputs: Summer 2018 daily LST and T2m (Version v0.1) [Dataset]. [Zenodo](http://doi.org/10.5281/zenodo.10806629)

## Contributing modeling software
| Model | Version | Author-Year | DOI |
|-------|---------|-----------------|-----|
 FVCOM | v41 | Chenfu Huang (2023a) | [Zenodo](http://doi.org/10.5281/zenodo.7574673)
 WRF | v4.2.2  | Chenfu Huang (2023b) | [Zenodo](http://doi.org/10.5281/zenodo.7574675)


## [Reproduce my experiment](workflow/README.md#reproduce-my-experiment)

## [Reproduce my analysis and figures](workflow/README.md#reproduce-my-analysis-and-figures)
