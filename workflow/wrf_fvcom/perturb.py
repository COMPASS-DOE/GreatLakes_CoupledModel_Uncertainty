import chaospy
import xarray as xr
import numpy as np
from enum import Enum
from os import PathLike
from typing import List, Union
from wrf_fvcom.variables import PerturbedVariable, VariableDistribution
from sklearn.preprocessing import OneHotEncoder
from skopt.space import Categorical, Real


class SampleRule(Enum):
    KOROBOV = 'korobov'
    RANDOM = 'random'
    SOBOL = 'sobol'
    LATINHYPERCUBE = 'latin_hypercube'


class TransformRule(Enum):
    ONEHOT = OneHotEncoder()


def distribution_from_variables(
    variables: List[PerturbedVariable], normalize: bool = False
) -> chaospy.Distribution:
    """
    :param variables: names of random variables we are perturbing
    :return: chaospy joint distribution encompassing variables
    """

    return chaospy.J(
        *(variable.chaospy_distribution(normalize=normalize) for variable in variables)
    )


def perturb_variables(
    variables: List[PerturbedVariable],
    number_perturbations: int = 1,
    sample_rule: SampleRule = SampleRule.RANDOM,
    output_directory: PathLike = None,
) -> xr.DataArray:
    """
    :param variables: names of random variables we are perturbing
    :param number_perturbations: number of perturbations for the ensemble
    :param sample_rule: rule for sampling the joint distribution (e.g., KOROBOV) see SampleRule class and chaospy docs
    :param output_directory: directory where to write the DataArray netcdf file, not written if None
    :return: DataArray of the perturbation_matrix
    """

    distribution = distribution_from_variables(variables)

    # get random samples from joint distribution
    random_sample = distribution.sample(number_perturbations, rule=sample_rule.value)
    if len(variables) == 1:
        random_sample = random_sample.reshape(-1, 1)
    else:
        random_sample = random_sample.T

    run_names = [
        f'{len(variables)}_variable_{sample_rule.value}_{index + 1}'
        for index in range(0, number_perturbations)
    ]
    variable_names = [f'{variable.name}' for variable in variables]

    perturbations = xr.DataArray(
        data=random_sample,
        coords={'run': run_names, 'variable': variable_names},
        dims=('run', 'variable'),
        name='perturbation_matrix',
    )

    if output_directory is not None:
        perturbations.to_netcdf(
            output_directory
            / f'perturbation_matrix_{len(variables)}variables_{sample_rule.value}{number_perturbations}.nc'
        )

    return perturbations


def transform_perturbation_matrix(
    perturbation_matrix: xr.DataArray,
    rule: TransformRule = TransformRule.ONEHOT,
    scale: bool = True,
    output_type: str = 'matrix',
) -> Union[xr.DataArray, List]:
    """
    :param perturbation_matrix: DataArray of the perturbation where categorical parameterizations are given in ordinal integers
    :param rule: rule for the transformation, see TransformRule class and sklearn preprocessing class. Only ONEHOT = OneHotEncoder() has been implemented.
    :param scale: scale non-categorical values to [0,1]?
    :param output_type:
        "matrix" - DataArray of transformed perturbation matrix, or
        "space"  - List of skopt.space types (Categorical or Real in range)
    :return: DataArray of the transformed perturbation_matrix
    """

    variable_names = perturbation_matrix['variable'].values
    runs = perturbation_matrix['run'].values
    # make the matrix for the categorical values
    categorical_matrix = []
    for run in runs:
        pvalues = perturbation_matrix.sel(run=run)
        variable_vector = []
        num_notcat_vars = 0
        for variable_name in variable_names:
            variable = PerturbedVariable.class_from_variable_name(variable_name)
            if variable.variable_distribution == VariableDistribution.DISCRETEUNIFORM:
                scheme_name = variable.return_scheme_name(pvalues.sel(variable=variable_name))
                variable_vector.append(scheme_name)
            else:
                num_notcat_vars += 1
        categorical_matrix.append(variable_vector)

    scheme_names = np.empty(0)
    variable_matrix = np.empty(0)
    space = []
    # transform the categorical values if there are any
    if len(categorical_matrix) > 0:
        enc = rule.value
        variable_matrix = enc.fit_transform(categorical_matrix).toarray()

        for schemes in enc.categories_:
            scheme_names = np.append(scheme_names, schemes)
            variable = PerturbedVariable.class_from_scheme_name(schemes[0])
            scheme_list = Categorical([scheme for scheme in schemes], name=variable.name)
            space.append(scheme_list)

    # now add on the non-categorial values if num_notcat_vars > 0
    if num_notcat_vars > 0:
        for variable_name in variable_names:
            variable = PerturbedVariable.class_from_variable_name(variable_name)
            if variable.variable_distribution != VariableDistribution.DISCRETEUNIFORM:
                pvalues = perturbation_matrix.sel(variable=variable_name).values
                if scale:
                    pvalues = (pvalues - variable.lower_bound) / (
                        variable.upper_bound - variable.lower_bound
                    )
                variable_matrix = np.append(variable_matrix, pvalues.reshape(-1, 1), axis=1)
                scheme_names = np.append(scheme_names, variable_name)
                parameter_range = Real(
                    variable.lower_bound, variable.upper_bound, name=variable.name
                )
                space.append(parameter_range)

    if output_type == 'matrix':
        return xr.DataArray(
            data=variable_matrix,
            coords={'run': runs, 'scheme': scheme_names},
            dims=('run', 'scheme'),
            name='transformed_perturbation_matrix',
        )
    elif output_type == 'space':
        return space
    else:
        raise ValueError(f'{output_type} not recognized. must be "matrix" or "space"')


def parameter_dict_to_perturbation_vector(param_dict: dict) -> xr.DataArray:
    """
    :param parameter_dict: Dictionary of parameterization schemes and parameters
    :return: DataArray of the perturbation_vector (integers and floats)
    """

    perturbation_vector = np.empty(len(param_dict))
    for vdx, var_name in enumerate(param_dict):
        variable = PerturbedVariable.class_from_variable_name(var_name)
        if variable.variable_distribution == VariableDistribution.DISCRETEUNIFORM:
            for value in range(variable.lower_bound, variable.upper_bound + 1):
                scheme_name = variable.return_scheme_name(value)
                if scheme_name == param_dict[var_name]:
                    perturbation_vector[vdx] = value
        else:
            perturbation_vector[vdx] = param_dict[var_name]

    return xr.DataArray(
        data=perturbation_vector,
        coords={'variable': list(param_dict.keys())},
        dims='variable',
        name='perturbation_vector',
    )
