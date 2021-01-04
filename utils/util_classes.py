import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.data.dataset
import torch.utils.data.dataset
import xarray as xr

from utils.util_args import multiclass_variables, partitions_dir, input_dir
from utils.util_funcs import create_k_folds_by_family, deconfound_dataset, set_attrs_from_parent_instance, \
    get_partitions_assigned_to_sets_in_fold, are_not_PD, find_nearest_PD_neighbor, transform_to_tangent, \
    onehot_to_multiclass, str_to_list


class ClassFromDict(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


class InputData(object):
    def __init__(self, params):
        self.n_folds = params.n_folds
        self.deconfound_flavor = params.deconfound_flavor
        self.scale_confounds = params.scale_confounds
        self.scale_features = params.scale_features
        self.confound_names = params.confound_names
        self.outcome_names = params.outcome_names
        self.subjects = params.ds.subject.values
        self.families_together = params.families_together
        self.matrix_directory = params.matrix_directory
        self.matrix_labels = str_to_list(params.matrix_labels)
        self.verbose = params.verbose
        self.n_subjects = len(self.subjects)
        self.saved_partitions_path = os.path.join(partitions_dir,
                                                  f'cv{self.n_folds}_{self.matrix_directory}_partitions.pkl')

        self.Family_IDs = self.get_family_ids(params)
        variables_of_interest = self.matrix_labels + self.outcome_names + list(filter(None, self.confound_names))
        if self.Family_IDs is not None:
            variables_of_interest.extend(['Family_ID'])
        self.ds = params.ds[variables_of_interest]
        self.partitioned_subjects = self.get_partitions()

    def load_existing_partitions(self):
        if os.path.isfile(self.saved_partitions_path):
            partitions = pickle.load(open(self.saved_partitions_path, "rb"))
        else:
            partitions = None
        return partitions

    def generate_partitions(self):
        subjects_by_fold, _ = create_k_folds_by_family(family_IDs=self.Family_IDs,
                                                       k_folds=self.n_folds,
                                                       keep_families_together=self.families_together,
                                                       shuffle=False)
        partitions = dict(zip(range(self.n_folds), subjects_by_fold))
        return partitions

    def save_partitions(self, partitions):
        pickle.dump(partitions, open(self.saved_partitions_path, "wb"))

    def get_partitions(self):
        partitioned_subjects = self.load_existing_partitions()
        if not partitioned_subjects:
            partitioned_subjects = self.generate_partitions()
            self.save_partitions(partitioned_subjects)
        return partitioned_subjects

    def get_family_ids(self, params):
        try:
            Family_IDs = params.ds[['Family_ID']]
        except KeyError:
            Family_IDs = None
        return Family_IDs


class XData(InputData):
    def __init__(self, params):
        InputData.__init__(self, params)
        self.xdata = self.ds[self.matrix_labels]
        self.confounds = [self.ds[cf].values for cf in self.confound_names] if any(self.confound_names) else None
        self.cardinality = int(self.xdata[self.matrix_labels[0]].dim1.__len__())
        self.n_features = int(((self.cardinality * self.cardinality) - self.cardinality) / 2)
        self.n_input = params.n_input
        self.multi_input = params.multi_input
        self.transformations = params.transformations
        self.tan_mean = params.tan_mean


class YData(InputData):
    def __init__(self, params):
        InputData.__init__(self, params)
        self.multiclass = self.outcome_names[0] in multiclass_variables
        self.n_classes = None
        self.class_weights = None
        self.ydata = None
        self.multioutcome = None
        self.n_outcomes = None

        self.set_ydata_as_array()
        self.set_outcome_attributes()

    def encode_ydata_as_onehot(self):
        import pandas as pd
        self.ydata = pd.get_dummies(self.ydata.squeeze(), dtype=float).to_numpy()

    def set_class_weights(self):
        self.class_weights = self.ydata.sum(axis=0) / len(self.ydata)  # class weighting (inverse of class frequency)

    def set_n_classes(self):
        self.n_classes = self.ydata.shape[1] if self.multiclass else 1

    def set_multiclass_outcome_names(self):
        self.outcome_names = [f'{self.outcome_names[0]}_{i}' for i in range(self.n_classes)]

    def set_ydata_as_array(self):
        self.ydata = self.ds[self.outcome_names].to_array().values.squeeze()

        if self.multiclass:
            self.encode_ydata_as_onehot()
            self.set_class_weights()
            self.set_n_classes()
            self.set_multiclass_outcome_names()

        if self.ydata.shape[0] != self.n_subjects:  # ensures 1st dimension is subject, 2nd is outcome
            self.ydata = self.ydata.T

    def set_outcome_attributes(self):
        self.multioutcome = self.ydata.shape.__len__() > 1
        self.n_outcomes = self.ydata.shape[-1] if self.multioutcome else 1

    def get_ydata_as_xarray(self, ydata):
        if self.multiclass or self.multioutcome:
            ydata = xr.DataArray(ydata, coords=dict(subject=self.subjects, outcome=self.outcome_names),
                                 dims=['subject', 'outcome'])
        else:
            ydata = xr.DataArray(ydata, coords=dict(subject=self.subjects), dims='subject')

        return ydata


class TransformedData(XData, YData):
    def __init__(self, fold, X, Y):
        set_attrs_from_parent_instance(self, X)
        set_attrs_from_parent_instance(self, Y)

        self.fold = fold
        self.transformed_xdata = self.xdata
        self.transformed_ydata = self.ydata
        self.transformed_matrix_labels = self.matrix_labels
        self.transformed_outcome_names = self.outcome_names
        self.fold_ds_path = f'{input_dir}/{self.matrix_directory}/{self.matrix_directory}_preprocessed_cv{self.n_folds}_fold{self.fold}.nc '
        self.saved_fold_ds = os.path.exists(self.fold_ds_path)

        self.partitions_assigned_to_sets = get_partitions_assigned_to_sets_in_fold(self.n_folds, self.fold)
        self.train_subjects = self.get_subjects_in_set_in_fold('train')
        _, self.train_subjects_inds, _ = np.intersect1d(self.subjects, self.train_subjects, return_indices=True)

        self.fold_preamble = f'cv{self.n_folds}_train{"".join([str(x) for x in self.partitions_assigned_to_sets["train"]])}'
        self.dec_preamble = '_'.join(
            ['dec', self.scale_confounds, *self.confound_names]) if any(self.confound_names) else None
        self.pd_preamble = 'pd' if 'tangent' in self.transformations else None
        self.tan_preamble = f'tan_{self.tan_mean}' if 'tangent' in self.transformations else None

        self.deconfounded_labels = None
        self.positive_definite_labels = None
        self.tangent_labels = None

        def transform_needed():
            needed = False if 'untransformed' in self.transformations or 'X0Y0' in self.deconfound_flavor else True
            return needed

        if transform_needed():
            foldwise_transformed_matrix_labels = ['_'.join([self.fold_preamble, x])
                                                  for x in self.matrix_labels]

            for i, label in enumerate(foldwise_transformed_matrix_labels):
                self.transformed_xdata = self.transformed_xdata.assign(
                    {label: self.transformed_xdata[self.matrix_labels[i]]})

            self.set_transformed_matrix_labels(foldwise_transformed_matrix_labels)

    def preprocess_data(self):
        if any([flavor in self.deconfound_flavor for flavor in ['X1Y1', 'X1Y0']]):
            if not self.are_deconfounded_saved() or 'X1Y1' in self.deconfound_flavor:
                self.encode_multiclass_confounds_as_onehot()
                if self.scale_confounds:
                    self.set_scaled_confounds()
                self.deconfound()
                self.save_to_xarray_ds()
            else:
                self.load_saved_deconfounded()

        if 'tangent' in self.transformations:
            if not self.are_tangent_saved():
                if not self.are_positive_definite_saved():
                    self.transform_to_positive_definite()
                else:
                    self.load_saved_positive_definite()
                self.transform_to_tangent()
                self.save_to_xarray_ds()
            else:
                self.load_saved_tangent()

        if self.scale_features:
            self.set_scaled_features()

        self.transformed_ydata = self.get_ydata_as_xarray(self.transformed_ydata)

    def set_deconfounded_labels(self):
        self.deconfounded_labels = ['_'.join([self.dec_preamble, x]) for x in self.transformed_matrix_labels]
        self.transformed_outcome_names = ['_'.join([self.dec_preamble, x]) for x in self.transformed_outcome_names]

    def set_positive_definite_labels(self):
        self.positive_definite_labels = ['_'.join([self.pd_preamble, x]) for x in self.transformed_matrix_labels]

    def set_tangent_labels(self):
        self.tangent_labels = ['_'.join([self.tan_preamble, x]) for x in self.transformed_matrix_labels]

    def set_transformed_matrix_labels(self, labels):
        self.transformed_matrix_labels = labels

    def are_deconfounded_saved(self):
        if not os.path.isfile(self.fold_ds_path):
            saved = False
            self.set_deconfounded_labels()
        else:
            saved_dataset = xr.open_dataset(self.fold_ds_path)
            self.set_deconfounded_labels()
            saved = True if all(
                [label in list(saved_dataset.data_vars) for label in self.deconfounded_labels]) else False
        return saved

    def are_tangent_saved(self):
        if not os.path.isfile(self.fold_ds_path):
            saved = False
            self.set_tangent_labels()
        else:
            saved_dataset = xr.open_dataset(self.fold_ds_path)
            self.set_tangent_labels()
            saved = True if all([label in list(saved_dataset.data_vars) for label in self.tangent_labels]) else False
        return saved

    def are_positive_definite_saved(self):
        if not os.path.isfile(self.fold_ds_path):
            saved = False
            self.set_positive_definite_labels()
        else:
            saved_dataset = xr.open_dataset(self.fold_ds_path)
            self.set_positive_definite_labels()
            saved = True if all(
                [label in list(saved_dataset.data_vars) for label in self.positive_definite_labels]) else False
        return saved

    def load_saved_deconfounded(self):
        saved_dataset = xr.open_dataset(self.fold_ds_path)
        self.transformed_xdata = saved_dataset[self.deconfounded_labels]
        self.set_transformed_matrix_labels(self.deconfounded_labels)

    def load_saved_positive_definite(self):
        saved_dataset = xr.open_dataset(self.fold_ds_path)
        self.transformed_xdata = saved_dataset[self.positive_definite_labels]
        self.set_transformed_matrix_labels(self.positive_definite_labels)

    def load_saved_tangent(self):
        saved_dataset = xr.open_dataset(self.fold_ds_path)
        self.transformed_xdata = saved_dataset[self.tangent_labels]
        self.set_transformed_matrix_labels(self.tangent_labels)

    def deconfound(self):
        for i, mtx_label in enumerate(self.transformed_matrix_labels):

            if self.verbose:
                if 'X1Y0' in self.deconfound_flavor:
                    print(f'\nDeconfounding {mtx_label} data using {", ".join(self.confound_names)} as confounds...')
                elif 'X1Y1' in self.deconfound_flavor:
                    print(
                        f'\nDeconfounding {mtx_label} data and {self.outcome_names}'
                        f' using {", ".join(self.confound_names)} as confounds...')

            mtx_data = self.transformed_xdata[mtx_label]
            dims = list(mtx_data.dims)
            X_dec, Y_dec, _ = deconfound_dataset(data=mtx_data.values,
                                                 confounds=self.confounds,
                                                 set_ind=self.train_subjects_inds,
                                                 outcome=self.transformed_ydata)

            self.transformed_xdata = self.transformed_xdata.assign(
                {self.deconfounded_labels[i]: xr.DataArray(data=X_dec,
                                                           coords=dict(zip(dims, [mtx_data[x].values for x in dims])),
                                                           dims=dims)})

            if 'X1Y1' in self.deconfound_flavor:
                self.transformed_ydata = Y_dec

        self.transformed_ydata = self.get_ydata_as_xarray(self.transformed_ydata)
        self.set_transformed_matrix_labels(self.deconfounded_labels)

    def transform_to_positive_definite(self):
        for i, mtx_label in enumerate(self.transformed_matrix_labels):

            mtx_data = self.transformed_xdata[mtx_label]
            notPD_count, _ = are_not_PD(mtx_data.values)

            if not notPD_count:
                self.transformed_xdata = self.transformed_xdata.assign({self.positive_definite_labels[i]: mtx_data})
            else:
                if self.verbose:
                    print(f'\nTransforming {mtx_label} data to positive definite...')
                self.transformed_xdata = self.transformed_xdata.assign(
                    {self.positive_definite_labels[i]:
                         xr.apply_ufunc(find_nearest_PD_neighbor, mtx_data.groupby('subject'))})

        self.set_transformed_matrix_labels(self.positive_definite_labels)

    def transform_to_tangent(self):
        for i, mtx_label in enumerate(self.transformed_matrix_labels):
            mtx_data = self.transformed_xdata[mtx_label]
            if self.verbose:
                print(f'\nTransforming {mtx_label} data to tangent space...')

            X_tan = transform_to_tangent(reference_matrices=mtx_data.loc[dict(subject=self.train_subjects)],
                                         projection_matrices=mtx_data.values,
                                         ref_mean=self.tan_mean)

            dims = list(mtx_data.dims)
            self.transformed_xdata = self.transformed_xdata.assign(
                {self.tangent_labels[i]: xr.DataArray(data=X_tan,
                                                      coords=dict(zip(dims, [mtx_data[x].values for x in dims])),
                                                      dims=dims)})

        self.set_transformed_matrix_labels(self.tangent_labels)

    def save_to_xarray_ds(self):
        if os.path.isfile(self.fold_ds_path):
            ds = xr.open_dataset(self.fold_ds_path)
            # note: overwrites previously transformed matrices with the same name
            ds = xr.merge([self.transformed_xdata, ds], compat='override', join='inner')
        else:
            ds = self.transformed_xdata
        ds.to_netcdf(self.fold_ds_path)

    def encode_multiclass_confounds_as_onehot(self):
        for i, confound_name in enumerate(self.confound_names):
            if confound_name in multiclass_variables:
                _, self.confounds[i] = np.unique(self.confounds[i], return_inverse=True)

    def set_scaled_confounds(self):
        self.confounds = [(x - np.min(x[self.train_subjects_inds])) /
                          (np.max(x[self.train_subjects_inds]) - np.min(x[self.train_subjects_inds]))
                          for x in self.confounds]

    def set_scaled_features(self):
        for label in self.transformed_matrix_labels:
            mtx_data = self.transformed_xdata[label]
            mtx_data_train = mtx_data.loc[dict(subject=self.train_subjects)]
            scaled_mtx_data = (mtx_data - mtx_data_train.min().values) / \
                              (mtx_data_train.max().values - mtx_data_train.min().values)

            self.transformed_xdata = self.transformed_xdata.assign({label: scaled_mtx_data})

    def get_oneD_data(self, subjects):
        oneD_X = self.get_unraveled_xdata(self.transformed_xdata, self.transformed_matrix_labels, subjects)
        oneD_Y = self.transformed_ydata.sel(dict(subject=subjects)).values

        if self.multiclass:
            oneD_Y = onehot_to_multiclass(oneD_Y)

        oneD_X = oneD_X.reshape(len(subjects), -1)
        oneD_Y = oneD_Y.reshape(len(subjects), -1)

        return oneD_X, oneD_Y

    def get_unraveled_xdata(self, xdata, matrix_labels, subjects):
        oneD_X = np.concatenate([xdata[var].sel(dict(subject=subjects)).values[:,
                                 np.triu_indices_from(xdata[var][0], k=1)[0],
                                 np.triu_indices_from(xdata[var][0], k=1)[1]]
                                 for var in matrix_labels], axis=1)

        return oneD_X

    def get_subjects_in_set_in_fold(self, set_name):
        import itertools
        set_partitions = self.partitions_assigned_to_sets[set_name]
        subjects = list(itertools.chain(*[self.partitioned_subjects[set_i] for set_i in set_partitions]))
        return subjects


class HCPDataset(torch.utils.data.Dataset):
    def __init__(self, transformed_data, mode="train", transform=False):
        super(HCPDataset, self).__init__()
        self.mode = mode
        self.transform = transform

        attrs = ['get_subjects_in_set_in_fold', 'transformed_matrix_labels', 'transformed_xdata',
                 'transformed_ydata', 'n_input', 'multi_input']
        set_attrs_from_parent_instance(self, transformed_data, attrs=attrs)

        self.mode_subjects = self.get_subjects_in_set_in_fold(self.mode)
        self.X = self.get_X_tensor()
        self.Y = self.get_Y_tensor()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = [self.X[idx], self.Y[idx]]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample

    def get_X_tensor(self):
        x = xr.merge([self.transformed_xdata[label].loc[dict(subject=self.mode_subjects)]
                      for label in self.transformed_matrix_labels]).to_array().values
        x = x.reshape(-1, self.n_input, x.shape[-1], x.shape[-1]).squeeze()
        if self.multi_input:
            x = torch.FloatTensor(x)
        else:
            x = torch.FloatTensor(np.expand_dims(x, 1))  # casting .astype(np.float64) may cause issues
        return x

    def get_Y_tensor(self):
        y = self.transformed_ydata.loc[dict(subject=self.mode_subjects)].values
        y = torch.FloatTensor(y)
        return y


class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, example, bias=False):
        super(E2EBlock, self).__init__()
        self.d = example.size(3)
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


class Edge2Edge(nn.Module):
    def __init__(self, channel, dim, filters):
        super(Edge2Edge, self).__init__()
        self.channel = channel
        self.dim = dim
        self.filters = filters
        self.row_conv = nn.Conv2d(channel, filters, (1, dim))
        self.col_conv = nn.Conv2d(channel, filters, (dim, 1))

    # implemented by two conv2d with line filter
    def forward(self, x):
        size = x.size()
        row = self.row_conv(x)
        col = self.col_conv(x)
        row_ex = row.expand(size[0], self.filters, self.dim, self.dim)
        col_ex = col.expand(size[0], self.filters, self.dim, self.dim)
        return row_ex + col_ex


class Edge2Node(nn.Module):
    def __init__(self, channel, dim, filters):
        super(Edge2Node, self).__init__()
        self.channel = channel
        self.dim = dim
        self.filters = filters
        self.row_conv = nn.Conv2d(channel, filters, (1, dim))
        self.col_conv = nn.Conv2d(channel, filters, (dim, 1))

    def forward(self, x):
        row = self.row_conv(x)
        col = self.col_conv(x)
        return row + col.permute(0, 1, 3, 2)


class Node2Graph(nn.Module):
    def __init__(self, channel, dim, filters):
        super(Node2Graph, self).__init__()
        self.channel = channel
        self.dim = dim
        self.filters = filters
        self.conv = nn.Conv2d(channel, filters, (dim, 1))

    def forward(self, x):
        return self.conv(x)


class HeSexBNCNN(torch.nn.Module, YData):
    """https://dx.doi.org/10.1101/473603"""

    def __init__(self, example, transformed_data):
        print('\nInitializing BNCNN: (He et al. 2018, Sex) BrainNetCNN architecture...')
        super(HeSexBNCNN, self).__init__()
        set_attrs_from_parent_instance(self, transformed_data, ['n_classes'])

        self.in_planes = example.size(1)
        self.d = example.size(3)

        self.e2econv1 = E2EBlock(example.size(1), 38, example, bias=True)
        self.E2N = torch.nn.Conv2d(38, 58, (1, self.d))
        self.N2G = torch.nn.Conv2d(58, 7, (self.d, 1))
        self.dense1 = torch.nn.Linear(7, self.n_classes)

        for m in self.modules():  # initializing weights
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = F.dropout(self.e2econv1(x), p=0.463)
        out = F.dropout(self.E2N(out), p=0.463)
        out = F.dropout(self.N2G(out), p=0.463)
        out = out.view(out.size(0), -1)
        out = torch.sigmoid(self.dense1(out))

        return out


class PervaizBNCNN(torch.nn.Module, YData):
    """https://doi.org/10.1016/j.neuroimage.2020.116604"""

    def __init__(self, example, transformed_data):
        print('\nInitializing (Pervaiz et al. 2020) BrainNetCNN architecture')
        super(PervaizBNCNN, self).__init__()
        set_attrs_from_parent_instance(self, transformed_data, ['n_classes', 'multiclass', 'n_outcomes'])
        self.in_planes = example.size(1)
        self.d = example.size(3)

        self.e2econv1 = E2EBlock(example.size(1), 32, example, bias=True)
        self.e2econv2 = E2EBlock(32, 64, example, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)  # init
        self.dense2 = torch.nn.Linear(128, 30)
        if self.multiclass:
            self.dense3 = torch.nn.Linear(30, self.n_classes)
        else:
            self.dense3 = torch.nn.Linear(30, self.n_outcomes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = F.dropout(F.leaky_relu(self.e2econv1(x), negative_slope=0.33), p=.5)
        out = F.dropout(F.leaky_relu(self.e2econv2(out), negative_slope=0.33), p=.5)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.relu(self.dense1(out)), p=0.5)
        out = F.dropout(F.relu(self.dense2(out)), p=0.5)

        if self.multiclass:
            out = torch.sigmoid(self.dense3(out))
        else:
            out = F.relu(self.dense3(out))

        return out


class KawaharaBNCNN(torch.nn.Module, YData):
    """https://doi.org/10.1016/j.neuroimage.2016.09.046"""

    def __init__(self, example, transformed_data):
        print('\nInitializing (Kawahara et al., 2016) BrainNetCNN architecture')
        super(KawaharaBNCNN, self).__init__()
        set_attrs_from_parent_instance(self, transformed_data, ['n_classes', 'multiclass', 'n_outcomes'])
        self.in_planes = example.size(1)
        self.d = example.size(3)

        self.e2econv1 = E2EBlock(1, 32, example, bias=True)
        self.e2econv2 = E2EBlock(32, 32, example, bias=True)
        self.E2N = torch.nn.Conv2d(32, 64, (1, self.d))
        self.N2G = torch.nn.Conv2d(64, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)
        self.dense2 = torch.nn.Linear(128, 30)
        self.batchnorm = torch.nn.BatchNorm1d(30)
        self.dense3 = torch.nn.Linear(30, self.n_outcomes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # taken from paper section 2.3 description
    def forward(self, x):
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.relu(self.dense1(out))
        out = F.dropout(F.relu(self.dense2(out)), p=0.5)
        out = F.relu(self.dense3(out))

        return out


class He58behaviorsBNCNN(torch.nn.Module, YData):
    """https://dx.doi.org/10.1101/473603"""

    def __init__(self, example, transformed_data):
        print('\nInitializing BNCNN: (He et al. 2018, 58 behaviors) BrainNetCNN architecture...')
        super(He58behaviorsBNCNN, self).__init__()
        set_attrs_from_parent_instance(self, transformed_data, ['n_classes', 'multiclass', 'n_outcomes'])
        self.in_planes = example.size(1)
        self.d = example.size(3)

        self.e2econv1 = E2EBlock(example.size(1), 18, example, bias=True)
        self.E2N = torch.nn.Conv2d(18, 19, (1, self.d))
        self.N2G = torch.nn.Conv2d(19, 84, (self.d, 1))

        if self.multiclass:
            self.dense1 = torch.nn.Linear(84, self.n_classes)
        else:
            self.dense1 = torch.nn.Linear(84, self.n_outcomes)

        for m in self.modules():  # initializing weights
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = F.dropout(self.e2econv1(x), p=0.463)
        out = F.dropout(self.E2N(out), p=0.463)
        out = F.dropout(self.N2G(out), p=0.463)
        out = out.view(out.size(0), -1)
        out = self.dense1(out)

        return out
