import datetime
import os
import pickle

import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from sklearn import neural_network, svm
from sklearn.linear_model import ElasticNet, SGDClassifier, MultiTaskElasticNet
from sklearn.metrics import balanced_accuracy_score, r2_score, mean_absolute_error

from utils.util_args import seed, performance_dir, decimals, p_thresh, test_sklearn_metrics, models_dir, \
    output_dir
from utils.util_classes import ClassFromDict, TransformedData
from utils.util_funcs import ensure_subfolder_in_folder, get_training_params


def main(args):
    params = ClassFromDict(args)
    X = params.X
    Y = params.Y
    rundate = datetime.datetime.now().strftime('%b_%d_%Y_%H_%M_%S')

    # note: train_{model}() functions only return test results, not train
    def train_SVM(trainx, trainy, testx=None, testy=None, results=None):
        print('\nTraining SVM...')

        if Y.multiclass:
            net = svm.SVC(kernel='linear', gamma='scale', verbose=params.verbose, random_state=seed,
                          class_weight='balanced')  # default reg. param C = 1.0
            net.fit(trainx, trainy)
            testp = net.predict(testx)
            trainp = net.predict(trainx)
            t_bacc = balanced_accuracy_score(testy, testp)

            results['test_balanced_accuracy'].append(t_bacc)

        else:
            net = svm.SVR(kernel='linear', gamma='scale', verbose=params.verbose)
            net.fit(trainx, trainy)
            testp = net.predict(testx)
            trainp = net.predict(trainx)
            t_r2 = r2_score(testy, testp)
            t_mae = mean_absolute_error(testy, testp)

            results['test_r2'].append(t_r2)
            results['test_mean_absolute_error'].append(t_mae)

        best_output = [trainp, trainy, testp, testy]
        output_names = ['trainp', 'trainy', 'testp', 'testy']

        return results, net, best_output, output_names

    def train_FC90net(trainx, trainy, testx=None, testy=None, results=None):
        print('\nTraining FC90Net...')

        if Y.multioutcome:
            fl = Y.n_outcomes
        elif Y.multiclass:
            fl = Y.n_classes
        else:
            fl = 1

        if Y.multiclass:
            hl_sizes = (3, fl)
            net = neural_network.MLPClassifier(hidden_layer_sizes=hl_sizes,
                                               max_iter=500,
                                               solver='sgd',
                                               learning_rate='adaptive',
                                               momentum=params.momentum,
                                               activation='relu',
                                               verbose=params.verbose,
                                               early_stopping=False,
                                               random_state=seed)

            net.fit(trainx, trainy)
            testp = net.predict(testx)
            trainp = net.predict(trainx)
            t_bacc = balanced_accuracy_score(testy, testp)

            results['test_balanced_accuracy'].append(t_bacc)

        else:
            hl_sizes = (9,)
            net = neural_network.MLPRegressor(hidden_layer_sizes=hl_sizes,
                                              max_iter=500,
                                              solver='sgd',
                                              learning_rate='adaptive',
                                              momentum=params.momentum,
                                              activation='relu',
                                              verbose=params.verbose,
                                              early_stopping=False,
                                              random_state=seed)

            net.fit(trainx, trainy)
            testp = net.predict(testx)
            trainp = net.predict(trainx)
            t_r2 = r2_score(testy, testp)
            t_mae = mean_absolute_error(testy, testp)

            results['test_r2'].append(t_r2)
            results['test_mean_absolute_error'].append(t_mae)

        best_output = [trainp, trainy, testp, testy]
        output_names = ['trainp', 'trainy', 'testp', 'testy']

        return results, net, best_output, output_names

    def train_ElasticNet(trainx, trainy, testx=None, testy=None, results=None):
        print('\nTraining ElasticNet...')

        if Y.multiclass:
            net = SGDClassifier(penalty='elasticnet', l1_ratio=.5,  # logistic regression with even L1/L2 penalty
                                random_state=seed)
            net.fit(trainx, trainy)
            testp = net.predict(testx)
            t_bacc = balanced_accuracy_score(testy, testp)

            results['test_balanced_accuracy'].append(t_bacc)

        elif Y.multioutcome:
            net = MultiTaskElasticNet(random_state=seed)
            net.fit(trainx, trainy)
            testp = net.predict(testx)
            trainp = net.predict(trainx)
            t_r2 = r2_score(testy, testp)
            t_mae = mean_absolute_error(testy, testp)

            results['test_r2'].append(t_r2)
            results['test_mean_absolute_error'].append(t_mae)

        else:
            net = ElasticNet(random_state=seed)
            net.fit(trainx, trainy)
            testp = net.predict(testx)
            trainp = net.predict(trainx)
            t_r2 = r2_score(testy, testp)
            t_mae = mean_absolute_error(testy, testp)

            results['test_r2'].append(t_r2)
            results['test_mean_absolute_error'].append(t_mae)

        best_output = [trainp, trainy, testp, testy]
        output_names = ['trainp', 'trainy', 'testp', 'testy']

        return results, net, best_output, output_names

    def train_CV(n_folds, train_func, scoring):
        """ Trains 1D nets with n_fold cross validation

        :param sklearn_cv: (bool) whether to use sklearn's cross validation implementation
        :param n_folds: (int) number of cross-validation folds to train over
        :param train_func: (function) 1D network training function in {train_SVM, train_FC90net, train_ElasticNet}
        """
        # identifiers for saving
        net_preamble = '_'.join([params.model[0], rundate])
        for folder in [performance_dir, models_dir, output_dir]:
            ensure_subfolder_in_folder(folder=folder, subfolder=params.model[0])

        features_keys = ['features_used']
        all_keys = scoring + features_keys
        cv_results = dict(zip(all_keys, [[] for _ in range(len(all_keys))]))

        for fold in range(n_folds):

            transformed_data = TransformedData(fold=fold, X=X, Y=Y)
            transformed_data.preprocess_data()
            train_subjects = transformed_data.get_subjects_in_set_in_fold('train') + \
                             transformed_data.get_subjects_in_set_in_fold('val')
            test_subjects = transformed_data.get_subjects_in_set_in_fold('test')

            X_train, Y_train = transformed_data.get_oneD_data(subjects=train_subjects)
            X_test, Y_test = transformed_data.get_oneD_data(subjects=test_subjects)

            def get_significant_features():
                if params.verbose:
                    print('Determining significant features in input data...')
                sig_features = np.zeros((transformed_data.n_features, transformed_data.n_outcomes), dtype=bool)
                for feature in range(transformed_data.n_features):
                    for outcome in range(transformed_data.n_outcomes):
                        _, p = pearsonr(X_train[:, feature], Y_train[:, outcome])
                        sig_features[feature, outcome] = bool(p < p_thresh)
                sig_features = np.sum(sig_features, axis=1)  # sig_features chosen as union over all outcomes
                assert sum(sig_features) > 0, 'No significant features detected'
                return sig_features

            features_to_use = get_significant_features() if params.use_sig_features \
                else np.ones(transformed_data.n_features, dtype=bool)

            X_train = X_train[:, features_to_use]
            X_test = X_test[:, features_to_use]
            Y_train = Y_train.squeeze()
            cv_results['features_used'].append(np.argwhere(features_to_use).squeeze().tolist())

            cv_results, net, best_output, output_names = train_func(trainx=X_train,
                                                                    trainy=Y_train,
                                                                    testx=X_test,
                                                                    testy=Y_test,
                                                                    results=cv_results)

            # saving net and best output
            net_path = os.path.join(models_dir, params.model[0], '_'.join([net_preamble + f'fold{fold}_net.pkl']))
            pickle.dump(net, open(net_path, "wb"))
            output_path = os.path.join(output_dir, params.model[0], '_'.join([net_preamble, f'fold{fold}_output.pkl']))
            pickle.dump(dict(zip(output_names, best_output)), open(output_path, "wb"))

        training_params = get_training_params(params=params, transformed_data=transformed_data)
        training_params.update({'rundate': rundate})

        return cv_results, training_params

    def save_CV_results(cv_results, training_params, scoring):
        """ Saves results of cross validated oneD network results

        :param scoring: results keys for performance metrics
        :param cv_results: the output of sklearn.model_selection.cross_validate()
        :return: None
        """

        model_name = training_params['model']
        n_folds = training_params['n_folds']

        performance_path = os.path.join(performance_dir, model_name)
        net_preamble = '_'.join([model_name, rundate])

        # separating info into dictionaries
        scoring_dict = {k: v for k, v in cv_results.items() if v and k in scoring}
        assert scoring_dict, 'results must have some keys in scoring'
        non_scoring_dict = {k: v for k, v in cv_results.items() if v and k not in scoring}

        metrics = list(scoring_dict.keys())  # performance metrics keys
        results_data = np.array(list(scoring_dict.values()))  # performance metrics data

        performance = xr.DataArray(results_data, coords=[metrics, range(n_folds)],
                                   dims=['metrics', 'cv_fold'], name=net_preamble)

        performance = performance.assign_attrs(training_params)

        # saving performance + partition and feature info
        performance.to_netcdf(os.path.join(performance_path, net_preamble + '_performance.nc'))
        if params.use_sig_features:
            ensure_subfolder_in_folder(os.path.join(performance_dir, params.model[0]), 'features_used')
            pickle.dump(non_scoring_dict, open(os.path.join(performance_path, 'features_used',
                                                            net_preamble + '_features_used.pkl'), "wb"))

    ensure_subfolder_in_folder(folder=performance_dir, subfolder=params.model[0])

    if 'SVM' in params.model:
        train_func = train_SVM
    elif 'FC90' in params.model:
        train_func = train_FC90net
    elif 'ElasticNet' in params.model:
        train_func = train_ElasticNet

    scoring = test_sklearn_metrics
    cv_results, attrs = train_CV(n_folds=params.n_folds, train_func=train_func, scoring=scoring)
    save_CV_results(cv_results, training_params=attrs, scoring=scoring)

    # # printing results
    np.set_printoptions(precision=decimals)
    print(f'\nResults, {params.model[0]} prediction of {params.outcome_names} from {params.matrix_labels}'
          f' over {params.n_folds} cross-validated folds:\n')
    for key, value in list(cv_results.items()):
        if value and key in scoring:
            print(f'{key}: {cv_results[key]}')


if __name__ == '__main__':
    main()
