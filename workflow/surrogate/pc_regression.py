import numpy as np
import chaospy
import logging
from wrf_fvcom.variables import PerturbedVariable
from wrf_fvcom.perturb import distribution_from_variables
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import (
    LassoCV,
    ElasticNetCV,
    LassoLarsCV,
    OrthogonalMatchingPursuit,
)
from numpoly import polynomial, ndpoly


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


def make_pc_surrogate_model(
    train_X, train_Y, polynomial_order: int = 1, regressor: str = 'Lasso', cv=LeaveOneOut(),
):

    nens, ndim = train_X.shape
    nens_, neig = train_Y.shape
    assert nens == nens_

    variable_transformed = [
        PerturbedVariable.class_from_scheme_name(scheme) for scheme in train_X['scheme']
    ]
    with DisableLogger():
        polynomial_expansion = chaospy.generate_expansion(
            order=polynomial_order,
            dist=distribution_from_variables(variable_transformed, normalize=True),
            rule='three_terms_recurrence',
        )

    if regressor == 'Lasso':
        reg = LassoCV(fit_intercept=False, cv=cv)
    elif regressor == 'ElasticNet':
        reg = ElasticNetCV(fit_intercept=False, cv=cv, l1_ratio=[.1, .5, .7, .9, .95, 1])
    elif regressor == 'Lars':
        reg = LassoLarsCV(fit_intercept=False, cv=cv)
        # reg = Lars(fit_intercept=False, n_nonzero_coefs=cv)
    elif regressor == 'OMP':
        # reg = OrthogonalMatchingPursuitCV(fit_intercept=False, cv=cv)
        reg = OrthogonalMatchingPursuit(fit_intercept=False, n_nonzero_coefs=cv)
    else:
        ValueError(f'{regressor} not recognized')

    poly_list = [None] * neig
    for mode in range(neig):
        train_yy = train_Y[:, mode]
        with DisableLogger():
            poly_list[mode] = chaospy.fit_regression(
                polynomials=polynomial_expansion,
                abscissas=train_X.T,
                evals=train_yy,
                model=reg,
            )

    surrogate_model = polynomial(poly_list)

    return surrogate_model
