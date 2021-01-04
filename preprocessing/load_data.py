import os

import xarray as xr

from utils.util_args import input_dir
from utils.util_classes import ClassFromDict, XData, YData
from utils.util_funcs import get_subject_info, read_mat_data


def main(args):
    params = ClassFromDict(args)
    read_in_anew = True
    xarray_path = f'{input_dir}/{params.matrix_directory}/{params.matrix_directory}.nc'
    saved = os.path.isfile(xarray_path)
    data_vars = [params.matrix_labels] if type(params.matrix_labels == str) else params.matrix_labels

    def read_ds_anew():
        ds = []
        for i, task_name in enumerate(params.tasks):

            if not task_name:
                task_name = params.matrix_directory
                name = params.matrix_labels
            else:
                name = params.matrix_labels[i]

            if params.verbose:
                print(f'\nReading in {task_name} data from directory {params.matrix_directory}...')

            partial, subjects = read_mat_data(f'{input_dir}/{params.matrix_directory}/{task_name}')
            nodes = [f'node_{x}' for x in range(partial.shape[-1])]

            partial = xr.DataArray(partial.squeeze(),
                                   coords=[subjects, nodes, nodes],
                                   dims=['subject', 'dim1', 'dim2'],
                                   name=name)

            ds.append(partial)

        ds = xr.align(*ds, join='inner')  # 'inner' takes intersection of ds objects
        ds = xr.merge(ds, compat='override', join='exact')

        return ds

    def load_ds():
        if saved:
            ds = xr.open_dataset(xarray_path)
            if not all([data_var in list(ds.data_vars) for data_var in data_vars]):
                ds = []
                read_in_anew = True
                if params.verbose:
                    print('Not all tasks found in saved .nc file. Reading data in anew...')
            else:
                ds = ds[data_vars]
                read_in_anew = False
                if params.verbose:
                    print('All tasks found in saved .nc file. Loading it...')

        if not saved or read_in_anew:
            ds = read_ds_anew()

        return ds

    ds = load_ds()
    ds = ds.sortby(['subject', 'dim1', 'dim2'], ascending=True)  # note: dim1/dim2 sorting here is (0,1,10,etc.)

    def add_subject_info_to_ds(ds):
        info_vars = params.outcome_names + list(filter(None, params.confound_names))
        if not all([var in list(ds.data_vars) for var in info_vars]):

            subject_info = get_subject_info(matrix_directory=params.matrix_directory,
                                            subjects=ds.subject.values).set_index('Subject')
            if 'Family_ID' in subject_info.columns:
                info_vars.extend(['Family_ID'])

            ds_subject_info = xr.Dataset(subject_info[info_vars]).rename_dims({'Subject': 'subject'})
            ds = xr.merge([ds, ds_subject_info], join='inner')
            # should drop subjects with any nan-valued subject info; important for deconfounding
            ds = ds.dropna(dim='subject')
            ds = ds.reset_coords('Subject', drop=True)

        return ds

    ds = add_subject_info_to_ds(ds)

    if read_in_anew:
        if saved:
            ds_saved = xr.open_dataset(xarray_path)
            ds = xr.merge([ds, ds_saved], compat='override', join='inner')
            ds.to_netcdf(xarray_path, mode='a')
        else:
            ds.to_netcdf(xarray_path)

    if params.verbose:
        print('\nMatrix, outcome, and confound data loaded...')

    args.update(dict(ds=ds))
    params = ClassFromDict(args)

    X = XData(params)
    Y = YData(params)

    return dict(X=X, Y=Y)


if __name__ == '__main__':
    main()
