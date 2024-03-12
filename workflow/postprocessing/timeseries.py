import numpy as np
import xarray as xr
from typing import List
from pandas import to_datetime, Timedelta, Timestamp
from datetime import datetime
from scipy.spatial import KDTree
from math import ceil


K2C = -273.15


def extract_timeseries_at_locations(
    files_lists: List, file_type: str, variable_name: str, lons, lats,
) -> List:
    """
    :param files_lists: list of list of files for each model run
    :param file_type: "wrf" cstm file or "fvcom" output file
    :param variable_name: variable name to extract
    :param lons: list of longitude locations
    :param lats: list of latitude locations
    :return: list of times and variables at the locations for each model run

   """

    if file_type == 'fvcom':
        times_per_file = 24
        vertical_layer = 0
        if variable_name == 'LST':
            variable_name = 'temp'
    elif file_type == 'wrf':
        times_per_file = 1
    else:
        raise ValueError(f'file_type {file_type} not recognized')

    output_list = []
    for file_list in files_lists:
        print(f'processing new model run, first file name: {file_list[0]}')
        wfv_value_at_locs = np.empty((len(file_list) * times_per_file, len(lons)))
        wfv_time = np.empty(len(file_list) * times_per_file, dtype='datetime64[s]')
        nn = 0
        for idx, fname in enumerate(file_list):
            if file_type == 'fvcom':
                wfv_temp = xr.open_dataset(
                    fname, decode_times=False, drop_variables=['siglay', 'siglev']
                )
                ddhh_vec = [
                    to_datetime('1858-11-17') + Timedelta(int(time_val * 24), 'h')
                    for time_val in wfv_temp.time.values
                ]
            elif file_type == 'wrf':
                wfv_temp = xr.open_dataset(fname)
                ddhh = datetime.fromisoformat(fname[-19::])

            if idx == 0:
                if file_type == 'fvcom':
                    wfv_grid_tree = KDTree(np.c_[wfv_temp.lon.values, wfv_temp.lat.values])
                elif file_type == 'wrf':
                    wfv_grid_tree = KDTree(
                        np.c_[
                            wfv_temp['XLONG'].values.flatten(),
                            wfv_temp['XLAT'].values.flatten(),
                        ]
                    )
                wfv_d, wfv_grid_ind = wfv_grid_tree.query(np.c_[lons, lats])

            if file_type == 'fvcom':
                value_temp = (
                    wfv_temp[variable_name]
                    .isel(siglay=vertical_layer, node=wfv_grid_ind)
                    .values
                )
                for ii, ddhh in enumerate(ddhh_vec):
                    wfv_time[nn] = np.datetime64(ddhh)
                    wfv_value_at_locs[nn, :] = value_temp[ii, :]
                    nn += 1
            elif file_type == 'wrf':
                wfv_value_at_locs[idx, :] = wfv_temp[variable_name].values.flatten()[
                    wfv_grid_ind
                ]
                wfv_time[idx] = np.datetime64(ddhh)
                nn += 1

        # remove non-unique values
        wfv_time, idx_start = np.unique(wfv_time[0:nn], return_index=True)
        wfv_value_at_locs = wfv_value_at_locs[idx_start, :]

        if variable_name == 'T2':
            wfv_value_at_locs += K2C

        output_list.append([wfv_time, wfv_value_at_locs])

    return output_list


def extract_daily_timeseries_global(
    files_lists: List, file_type: str, variable_name: str,
) -> List:

    if file_type == 'fvcom':
        days_per_file = 1
        vertical_layer = 0
        if variable_name == 'LST':
            variable_name = 'temp'
    elif file_type == 'wrf':
        days_per_file = 1 / 24
    else:
        raise ValueError(f'file_type {file_type} not recognized')

    output_list = []
    for file_list in files_lists:
        print(f'processing new model run, first file name: {file_list[0]}')
        wfv_time = np.empty(ceil(len(file_list) * days_per_file), dtype='datetime64[s]')
        nn = 0
        for idx, fname in enumerate(file_list):
            if file_type == 'fvcom':
                wfv_temp = xr.open_dataset(
                    fname, decode_times=False, drop_variables=['siglay', 'siglev']
                )
                wfv_time[nn] = to_datetime('1858-11-17') + Timedelta(
                    int(wfv_temp.time.values[0] * 24), 'h'
                )
            elif file_type == 'wrf':
                wfv_temp = xr.open_dataset(fname)
                ddhh = datetime.fromisoformat(fname[-19::])
                if ddhh.hour == 0:
                    wfv_time[nn] = np.datetime64(ddhh)

            if file_type == 'fvcom':
                value_temp = (
                    wfv_temp[variable_name].isel(siglay=vertical_layer).mean(dim='time')
                )
                if idx == 0:
                    wfv_daily_values = value_temp
                else:
                    wfv_daily_values = xr.concat([wfv_daily_values, value_temp], 'time')
                nn += 1
            elif file_type == 'wrf':
                if ddhh.hour == 0:
                    value_temp = wfv_temp[variable_name]
                else:
                    value_temp = xr.concat([value_temp, wfv_temp[variable_name]], dim='Time')
                if ddhh.hour == 23:
                    value_temp = value_temp.mean(dim='Time')
                    if nn == 0:
                        wfv_daily_values = value_temp
                    else:
                        wfv_daily_values = xr.concat([wfv_daily_values, value_temp], 'time')
                    nn += 1

        # remove non-unique values
        wfv_time, idx_start = np.unique(wfv_time[0:nn], return_index=True)
        wfv_daily_values = wfv_daily_values.isel(time=idx_start)

        # get the coordinates back
        if file_type == 'wrf':
            wfv_daily_values = wfv_daily_values.assign_coords(
                {
                    'XLONG': wfv_temp['XLONG'].isel(Time=0),
                    'XLAT': wfv_temp['XLAT'].isel(Time=0),
                }
            )

        if variable_name == 'T2':
            wfv_daily_values += K2C

        output_list.append([wfv_time, wfv_daily_values])

    return output_list


def align_model_with_observations(model: dict, data: List, remove_noise: bool = True) -> dict:
    """
    :param model: dictionary of model runs
    :param data: list of observation data
    :param remove noise: choice to remove noisy 2*dx waves in model data
    :return: dictionary of processes timeseries where data and model times overlap and shaped into a vector

   """

    # make the vectors from the model ensemble didct and buoy data list
    variables = list(model.keys())
    variables.remove('runs')

    time_vector = np.empty((0), dtype='datetime64')
    data_vector = {}
    model_vector = {}
    earliest_end_time = model[variables[0]][0][0][-1]
    for variable in variables:
        data_vector[variable] = np.empty((0), dtype='float')
        model_vector[variable] = np.empty((len(model['runs']), 0), dtype='float')
        # check earliest model end time
        for this_run in model[variable]:
            earliest_end_time = min(earliest_end_time, this_run[0][-1])
    print(f'earliest model end time: {earliest_end_time}')

    for bdx, this_loc in enumerate(data):
        mask = this_loc['data']['time'] <= earliest_end_time
        obs_time = this_loc['data']['time'][mask]
        time_vector = np.append(time_vector, obs_time)
        for variable in variables:
            # print(variable)
            data_vector[variable] = np.append(
                data_vector[variable], this_loc['data'][variable][mask]
            )

            model_temp = np.empty((len(model['runs']), len(obs_time)))
            for rr, this_run in enumerate(model[variable]):
                # print(rr)
                _, _, mod_int_ind = np.intersect1d(obs_time, this_run[0], return_indices=True)
                if remove_noise:
                    model_temp[rr, :] = remove_2dx_waves(this_run[1][mod_int_ind, bdx])
                else:
                    model_temp[rr, :] = this_run[1][mod_int_ind, bdx]
            model_vector[variable] = np.append(model_vector[variable], model_temp, axis=1)

    return {'time': time_vector, 'data': data_vector, 'model': model_vector}


def remove_2dx_waves(ts):
    # removing noisy 2*delta(X) waves
    bad_tol = 0.5 * np.sqrt(2) * np.std(ts)
    for tdx, tss in enumerate(ts):
        if tdx == 0 or tdx == ts.shape[0] - 1:
            continue
        tn = ts[tdx + 1]
        tp = ts[tdx - 1]
        if abs(tp - tss) > bad_tol and abs(tn - tss) > bad_tol:
            ts[tdx] = 0.5 * (tp + tn)

    return ts
