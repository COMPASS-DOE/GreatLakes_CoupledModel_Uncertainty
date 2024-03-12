from numpoly import ndpoly
from numpy import (
    stack,
    sum,
    sqrt,
    dot,
    empty,
    median
)


# for optional transformation of outputs prior to sensitivity sampling
def inverse_kl(eigenmodes, eigenvalues, samples, mean_vector):
    neig = eigenvalues.shape[0]
    if samples.ndim == 1:
       samples = samples.reshape(1,-1)
    y_out = sum(
        stack(
            [
                dot(
                    (samples * sqrt(eigenvalues))[:, mode_index, None],
                    eigenmodes[None, mode_index, :],
                )
                for mode_index in range(neig)
            ],
            axis=0,
        ),
        axis=0,
    )
    y_out += mean_vector
    return y_out


def surrogate_model_predict(surrogate_model, X_values, kl_dict=None):
    if type(surrogate_model) is ndpoly:
        Y_values = surrogate_model(*X_values.T).T
    elif type(surrogate_model) == list:
        n_folds = len(surrogate_model)
        for sdx, sm in enumerate(surrogate_model):
            #Y_values = sm.predict(X_values)
            if sdx == 0:
                Y_values = sm.predict(X_values)
                #YY_values = empty(Y_values.shape + (n_folds,))
            else:
                Y_values += sm.predict(X_values)
            #YY_values[:, :, sdx] = Y_values
        #Y_values = median(YY_values,axis=-1)
        Y_values /= n_folds
    else:
        Y_values = surrogate_model.predict(X_values)

    if kl_dict is not None:
        Y_values = inverse_kl(
            kl_dict['eigenmodes'], kl_dict['eigenvalues'], Y_values, kl_dict['mean_vector'],
        )

    return Y_values
