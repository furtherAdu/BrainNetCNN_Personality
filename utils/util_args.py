multiclass_variables = ['Gender']  # update with categorical headers in subject_info.csv (e.g. 'Gender')

figures_dir = 'figures'
input_dir = 'input_data'
models_dir = 'models'
partitions_dir = 'partitions'
performance_dir = 'performance'
sub_info_dir = 'subject_info'
output_dir = 'best_output'

seed = 1234
decimals = 2  # decimals to print
model_names = ['BNCNN', 'SVM', 'FC90', 'ElasticNet']
set_names = ['train', 'test', 'val']
metrics = ['loss', 'accuracy', 'MAE', 'pearsonr', 'pearsonp', 'spearmanr', 'spearmanp']
p_thresh = .001  # p-value threshold for determination of significant input data features before oneD training

# # figure headers for print_nested_results.py
date_header = ['rundate']
nested_headers = ['model', 'architecture', 'outcomes', 'input_data', 'transforms', 'best_test_epoch', 'set']

# performance_headers should include names all metrics for available models
test_sklearn_metrics = ['test_neg_mean_absolute_error', 'test_mean_absolute_error', 'test_r2', 'test_balanced_accuracy']
torch_metrics = ['MAE', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p', 'accuracy']
mean_metrics = [f'mean_{metric}' for metric in torch_metrics + test_sklearn_metrics]
performance_headers = torch_metrics + mean_metrics + test_sklearn_metrics
