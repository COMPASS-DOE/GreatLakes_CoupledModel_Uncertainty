from pathlib import Path
from wrf_fvcom.variables import (
    PerturbedVariable,
    WRF_PBL_SFCLAY,
    WRF_WaterZ0,
    WRF_MP,
    WRF_RA,
    WRF_LM,
    FVCOM_Prandtl,
    FVCOM_SWRadiationAbsorption,
    FVCOM_VerticalMixing,
    FVCOM_WindStress,
)
from wrf_fvcom.perturb import (
    perturb_variables,
    SampleRule,
    transform_perturbation_matrix,
)

if __name__ == '__main__':

    output_directory = Path.cwd().parent / 'output'

    variables = [
        WRF_PBL_SFCLAY,
        WRF_WaterZ0,
        WRF_MP,
        WRF_RA,
        WRF_LM,
        FVCOM_WindStress,
        FVCOM_VerticalMixing,
        FVCOM_Prandtl,
        FVCOM_SWRadiationAbsorption,
    ]

    number_perturbations = 18
    sample_rule = SampleRule.KOROBOV

    # calling the perturb_variables function (output is written inside the function)
    perturbations = perturb_variables(
        variables=variables,
        number_perturbations=number_perturbations,
        sample_rule=sample_rule,
        output_directory=output_directory,
    )

    # Dependent variables to follow...

    # Shortwave radiation absorption scheme
    # Calculate Z1 and Z2 from R fraction
    SW_R = perturbations.sel(variable=FVCOM_SWRadiationAbsorption.name)
    Z1 = FVCOM_SWRadiationAbsorption.calc_Z1(SW_R)
    Z2 = FVCOM_SWRadiationAbsorption.calc_Z2(SW_R)

    if output_directory is not None:
        file_prefix = f'perturbation_vector_{sample_rule.value}{number_perturbations}_FVCOM_SWRadiationAbsorption'
        Z1.to_netcdf(output_directory / f'{file_prefix}_Z1.nc')
        Z2.to_netcdf(output_directory / f'{file_prefix}_Z2.nc')

    # demonstrating the transformation of perturbations using OneHotEncoding
    variable_matrix = transform_perturbation_matrix(perturbations)

    # demonstrating retrieval of variable class from scheme names
    for scheme in variable_matrix['scheme']:
        variable_class = PerturbedVariable.class_from_scheme_name(scheme)

    # demonstrating output of transformed perturbations in list of skopt.space types
    variable_space = transform_perturbation_matrix(perturbations, output_type='space')
