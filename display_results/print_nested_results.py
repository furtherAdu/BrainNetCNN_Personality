import datetime
import os
import sys

sys.path.append(".")

from utils.util_funcs import namestr, performance_to_csv, csv_results_to_html
from utils.util_args import performance_headers, nested_headers, date_header, figures_dir, performance_dir, model_names

time_now = datetime.datetime.now().strftime("%B_%d_%Y")
performance_csv_path = os.path.join(performance_dir, f'csv_results_{time_now}.csv')
performance_html_path = os.path.join(figures_dir, f'nested_model_performance_{time_now}.html')
headers = [performance_headers, nested_headers, date_header]
headers = dict(zip([namestr(x, globals()) for x in headers], headers))

performance_to_csv(performance_csv_path, models=model_names)
csv_results_to_html(performance_csv_path, performance_html_path, headers=headers)
