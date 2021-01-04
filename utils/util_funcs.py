from __future__ import print_function

import os
import random

import numpy as np
import pandas as pd
import xarray as xr
from numpy import linalg as la
from scipy.linalg import logm, inv
from sklearn.model_selection import KFold

from utils.util_args import seed, set_names, performance_dir, decimals, input_dir, sub_info_dir


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]


def str_to_list(x):
    x_list = [x] if type(x) is str else x
    return x_list


def onehot_to_multiclass(Y):
    return np.array([np.where(i == 1)[0][0] for i in Y])


def get_subject_info(matrix_directory, subjects=None):
    info_csv = pd.read_csv(f'{input_dir}/{sub_info_dir}/{matrix_directory}_subject_info.csv')

    if any(subjects):
        subject_indices = np.where(np.isin(info_csv["Subject"], subjects))[0]
        info_csv = info_csv.reindex(subject_indices)  # data from only specified subjects

    return info_csv


def get_partitions_assigned_to_sets_in_fold(n_folds, current_fold):
    idxs_by_set = [range(n_folds - 2), [-2], [-1]]
    ordered_partitions_by_fold = (np.arange(n_folds) + current_fold) % n_folds
    return dict(zip(set_names, [ordered_partitions_by_fold[set_idxs] for set_idxs in idxs_by_set]))


def read_mat_data(mat_dir):
    """Reads in matrix data from a data directory.

    :param mat_dir: (filepath) data directory
    :return: (array) matrix data, (array) subject IDs
    """
    filenames = [f for f in os.listdir(mat_dir) if os.path.isfile(os.path.join(mat_dir, f)) and f.endswith('.txt')]
    filenames.sort()

    data = np.array([np.loadtxt(os.path.join(mat_dir, file)) for file in filenames], dtype=float)
    subjects = np.array([file.split('.txt')[0] for file in filenames], dtype=int)

    return data, subjects


def is_PD(B):
    """Returns true when input is positive-definite, via Cholesky
    credit: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


def are_not_PD(matrices):
    """
    Script to test many matrices for positive definiteness
    :param matrices: array of matrices
    :return: number of matrices that aren't PD, and their indices in manyB
    """
    count = 0
    notPD_indices = []
    for i, matrix in enumerate(matrices):
        if not is_PD(matrix):
            count += 1
            notPD_indices.append(i)

    return count, notPD_indices


def find_nearest_PD_neighbor(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    if is_PD(A):
        return A

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_PD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not is_PD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def transform_to_tangent(reference_matrices, projection_matrices, ref_mean='euclidean'):
    """
    Projects array of matrices (projection_matrices) into tangent space,
      using the mean of another array (refmats) as reference.

    Implementation from (Dadi et al., 2019, doi: 10.1016/j.neuroimage.2019.02.062)
    Calculation of reference means from (Pervaiz et al., 2020)

    :param reference_matrices: positive definite covariance matrices (samples x rows x columns), from which mean is calculated
    :param projection_matrices: positive definite matrices to be projected into tangent space
    :param ref_mean: reference mean to use (i.e. euclidean, harmonic, log euclidean, riemannian, kullback)
    :return: tangent-projected matrices
    """
    if ref_mean == 'harmonic':
        Ch = 0
        for i, x in enumerate(reference_matrices):
            Ch += inv(x)
        Ch *= 1 / len(reference_matrices)
        refMean = inv(Ch)

    elif ref_mean == 'euclidean':
        refMean = 1 / len(reference_matrices) * np.mean(reference_matrices, axis=0)

    d, V = np.linalg.eigh(refMean)  # EVD on reference mean covariance matrix
    fudge = 1E-18  # ensures our eigenvectors don't explode
    wsStar = V.T @ np.diag(1 / np.sqrt(d + fudge)) @ V
    cardinality = len(projection_matrices[1])

    tmats = np.zeros_like(projection_matrices)

    for i, x in enumerate(projection_matrices):
        m = np.dot(wsStar, x).dot(wsStar)
        m = m.reshape(cardinality, cardinality)
        tmats[i] = logm(m)

    return tmats


def get_confound_parameters(est_data, confounds, set_ind=None):
    """Takes array of square matrices (samples x matrices) and returns confound signals, the parameter.

    est_data: full data from which the confound parameters are estimated
    set_ind: indices of the est_data from which the confound parameters will be estimated
    confounds: list of confounds, each containing same number of samples as est_data
    data_tbd: data to be deconfounded

    return:
        nan_ind: the indices (out of the set_ind) that have any confound == nan
        C: the nan-removed confound matrix
        C_pi: pseudoinverse of confounds
        b_hatX: deconfounded X

    Calculations based off equations (2) - (4):
    https://www.sciencedirect.com/science/article/pii/S1053811918319463?via%3Dihub#sec2
    """

    # vectorizing matrix and subtracting mean
    t = np.array([x[np.triu_indices(len(x), k=1)] for x in est_data])
    t -= np.mean(t, axis=0)

    est_array = np.array([t[j] for j in list(set_ind)])  # specifying arrays from which we'll deconfound

    # creating confound matrix
    C = np.vstack(confounds).astype(float).T[set_ind]

    # identifying nan values in confounds
    nan_ind = np.unique(np.argwhere(np.isnan(C)).squeeze())

    # deleting samples that have confounds with NaN values
    C = np.delete(C, nan_ind, axis=0)
    X = np.delete(est_array, nan_ind, axis=0)

    # regressing out confounds
    C_pi = np.linalg.pinv(C)  # moore-penrose pseudoinverse
    b_hatX = C_pi @ X  # confound parameter estimate

    return C_pi, b_hatX, nan_ind


def reshape_array_as_matrix(samples, matrix_length):
    """
    :param samples: samples x upper triangular array of a matrix
    :param matrix_length: determined size of newly shaped matrix
    :return: mat_size x mat_size symmetric matrix
    """
    all_sample_matrices = []

    for i in range(len(samples)):  # reshaping array into symmetric matrix
        matrix = np.zeros((matrix_length, matrix_length))

        upper_triangle_indices = np.triu_indices(len(matrix), k=1)
        matrix[upper_triangle_indices] = samples[i]
        matrix = np.triu(matrix, 1) + matrix.T

        where_are_NaNs = np.isnan(matrix)  # changing error-prone NaN values to zero
        if np.any(where_are_NaNs):
            print(f'Setting NaNs to zero in matrix {i}...')
            matrix[where_are_NaNs] = 0

        all_sample_matrices.append(matrix)

    all_sample_matrices = np.array(all_sample_matrices)  # setting as array

    return all_sample_matrices


def deconfound_dataset(data, confounds, set_ind, outcome):
    """
    Takes input of a data, its confounds. Deletes samples with nan-valued Y entries.
     Returns the deconfounded data.

    :param outcome: ground truth value to be deconfounded
    :param data: Samples x symmetric matrices (row x column) to be deconfounded
    :param confounds: Confounds x samples, to be factored out of ds
    :param set_ind: sample indices of data from which deconfounding parameters will be calculated
    :return: List of deconfounded X, Y as well as new train-test-validation indices
    """

    # confound parameter estimation for X
    C_pi, b_hat_X, nan_ind = get_confound_parameters(data, confounds, set_ind=set_ind)

    # ...and Y, with nans removed
    Y_c = np.delete(outcome[set_ind], nan_ind, axis=0)
    b_hat_Y = C_pi @ Y_c  # Y confound parameter estimation

    C_tbd = np.vstack(confounds).astype(float).T

    X_dec = data - reshape_array_as_matrix(C_tbd @ b_hat_X, matrix_length=data.shape[-1])
    Y_dec = outcome - C_tbd @ b_hat_Y

    return np.array(X_dec), np.array(Y_dec), nan_ind


def create_k_folds_by_family(family_IDs, k_folds=6, keep_families_together=False, shuffle=False, seed=seed):
    """
    :param seed: random seed
    :param shuffle: shuffles order of families (v.s. loading from largest to smallest)
                Note: shuffling increases likelhood of folds of unequal sizes
    :param family_IDs: (xarray) dataset of Family_IDs, with subject numbers as coords
    :param k_folds: number of folds in which to partition the data
    :param keep_families_together: (bool) whether to keep families in the same fold
    :return: subjects in each fold, their indices in the data
    """

    subject_IDs = family_IDs.subject.values

    if family_IDs and keep_families_together:
        max_fold_size = np.ceil(len(subject_IDs) / k_folds)
        min_fold_size = np.floor(len(subject_IDs) / k_folds)
        remaining = np.remainder(len(subject_IDs), k_folds)

        families = family_IDs.groupby('Family_ID')._group_indices
        families.sort(key=len, reverse=True)  # biggest to smallest

        if shuffle:
            random.shuffle(families)  # shuffling order

        inds_in_fold = [[] for _ in range(k_folds)]
        counter = 0

        while counter < len(families):
            try:
                for i in range(k_folds):
                    added_inds = families[counter]

                    # pass over max full folds
                    if len(inds_in_fold[i]) == max_fold_size:
                        continue

                    # pass over min full folds if nothing remains
                    elif (len(inds_in_fold[i]) >= min_fold_size) and (remaining == 0):
                        continue

                    # add to folds until they are full
                    else:
                        inds_in_fold[i].extend(added_inds)
                        counter += 1

                        # if a fold exceeds min full, detract the excess from remaining
                        if len(inds_in_fold[i]) > min_fold_size:
                            remaining -= len(inds_in_fold[i]) - min_fold_size

                            # if the excess is too much, sum zero
                            if remaining < 0:
                                remaining = 0

            except IndexError:
                break

        inds_in_fold = np.array(inds_in_fold)

    else:  # tear families apart
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        inds_in_fold = np.array(list(kf.split(subject_IDs)))[:, 1]

    subs_in_fold = np.array([subject_IDs[x] for x in inds_in_fold])

    return subs_in_fold, inds_in_fold


def csv_results_to_html(performance_csv, performance_html, headers):
    """Creates html performance table from csv file of results"""

    df = pd.read_csv(performance_csv)
    performance_headers = [x for x in headers['performance_headers'] if x in list(df.columns)]
    nest_headers = [x for x in headers['nested_headers'] if x in list(df.columns)]
    date_header = [x for x in headers['date_header'] if x in list(df.columns)]
    available_headers = performance_headers + nest_headers + date_header

    df = df[available_headers]
    index = pd.MultiIndex.from_arrays(df[nest_headers].values.T, names=nest_headers)

    leveled_df = pd.DataFrame(df[performance_headers + date_header].values, index=index,
                              columns=performance_headers + date_header)
    leveled_df.sort_index(level=0, ascending=True, inplace=True)

    leveled_df.to_html(performance_html, bold_rows=True, na_rep='')


def performance_to_csv(performance_csv, models=None):
    """Compiles results of nested k-fold cross validation trainings."""
    results_df = pd.DataFrame()  # allocating dataframe for results
    partitions = ['test', 'val']

    for model in models:
        model_dir = os.path.join(performance_dir, model)
        if not os.path.exists(model_dir):
            continue
        filenames = [os.path.join(model_dir, y) for y in list(filter(lambda x: x.endswith('performance.nc'),
                                                                     os.listdir(model_dir)))]

        for filename in filenames:
            performance = xr.open_dataarray(filename)
            attrs = performance.attrs
            outcomes = [val for key, val in attrs.items() if key.startswith('outcome')]
            matrix_labels = [val for key, val in attrs.items() if key.startswith('matrix_label')]

            if len(outcomes) == 1:
                outcomes = outcomes[0]

            if model == 'BNCNN':
                folds = [int(key.split('best_test_epoch_fold_')[-1]) for key, val in attrs.items() if
                         key.startswith('best_test_epoch_fold_')]
                best_test_epochs = [val for key, val in attrs.items() if key.startswith('best_test_epoch_fold_')]
                best_mean_test_epoch = int(np.nanmean(best_test_epochs))

            for part in partitions:
                # creating dict of necessary performance info, keys should be in headers
                results_dict = dict(rundate=performance.rundate, input_data=matrix_labels, model=model,
                                    outcomes=outcomes, transforms=performance.transformations, set=part)

                for metric in list(performance.metrics.values):

                    if model == 'BNCNN':
                        results_dict.update({'best_test_epochs': best_test_epochs,
                                             'best_mean_test_epoch': best_mean_test_epoch})
                        entry = [performance.loc[dict(metrics=metric, cv_fold=folds[i], epoch=best, set=part)].values
                                 for i, best in enumerate(best_test_epochs)]
                    else:
                        entry = [performance.loc[dict(metrics=metric, cv_fold=fold)].values
                                 for fold in range(performance.n_folds)]

                    mean_entry = np.nanmean(entry)
                    results_dict.update({metric: np.array(entry).squeeze().round(decimals),
                                         f'mean_{metric}': mean_entry.squeeze().round(decimals)})

                results_df = results_df.append(results_dict, ignore_index=True)  # appending single run to dataframe

    results_df.to_csv(performance_csv)


def ensure_subfolder_in_folder(folder, subfolder):
    cwd = os.getcwd()
    subfolder_path = os.path.join(cwd, folder, subfolder)
    if not os.path.exists(subfolder_path):
        os.mkdir(subfolder_path)


def set_attrs_from_parent_instance(obj, base, attrs=[]):
    if not any(attrs):
        attrs = [attr for attr in dir(base) if not attr.startswith('__')]
    for attr in attrs:
        setattr(obj, attr, getattr(base, attr))


def get_training_params(params, transformed_data):
    training_params = dict(model=params.model[0],
                           architecture=params.architecture,
                           multiclass=transformed_data.multiclass,
                           multioutcome=transformed_data.multioutcome,
                           transformations=transformed_data.transformations,
                           deconfound_flavor=transformed_data.deconfound_flavor,
                           n_folds=transformed_data.n_folds,
                           scale_confounds=transformed_data.scale_confounds,
                           scale_features=transformed_data.scale_features,
                           early_stopping=params.early)

    for i, matrix_label in enumerate(transformed_data.matrix_labels):
        training_params.update({f'matrix_label_{i}': matrix_label})

    for i, outcome_name in enumerate(transformed_data.outcome_names):
        training_params.update({f'outcome_name_{i}': outcome_name})

    if any(transformed_data.confound_names):
        for i, confound_name in enumerate(transformed_data.confound_names):
            training_params.update({f'confound_name_{i}': confound_name})

    for key, val in training_params.items():
        if val is None:
            training_params[key] = 0

    return training_params
