import os, time
import csv
import enum
import string
import arguments.setting as setting
import datastructure.dataset as dataset
import learning.train_new as train_new
# import  learning.train_case_3 as train_new

# from learning.construct_features import construct_features
from progress.bar import IncrementalBar
# import learning.train
# import learning.test
import getopt, sys

def run(data, setting, log_dir, runs_start=0, runs=10):
    """ Run model training and testing

    Parameters
    ----------
    data : Dataset
        Dataset to use for model training/testing
    setting : Setting
        Setting as specified by class
    train : bool
        True if want to train models
    features_only : bool
        True if only want to encode features
    runs_stat : int
        First run of Monte-Carlo cross-validation to run
    runs : int
        Last run of Monte-Carlo cross-validation to run
    draw_map : bool
        True if want to save attention maps
    """
    if runs_start >= runs:
        return 
    import numpy as np

    print(np.shape(data.train_set))
    print("_-----")


    bar = IncrementalBar('Running Monte Carlo ', max=runs)

    # @MPR
    # Iterate Monte-carlo
    for k in range(runs_start, runs):
        fold_log_dir = os.path.join(log_dir, f'fold_{k}')
        os.makedirs(fold_log_dir)
        # Set split
        data.set_fold(k)
        # Train model
        train_new.train(data.get_train_set(), data.get_validation_set(), k, fold_log_dir, setting)
        # Test model
        train_new.test(data.get_test_set(), k, fold_log_dir, setting)
        bar.next()

    bar.finish()

# @MPR
def run_train(data_directories, csv_file, working_directory, qupath_project_file):
    """ Set up setting and dataset and run training/testing
    """
    s = setting.Setting(data_directories, csv_file, working_directory, qupath_project_file)

    data = dataset.Dataset(s)
    log_dir = os.path.join(working_directory, 'Runs', f'{s.get_network_setting().get_run_identifier()}-{time.strftime("%y-%m-%d-%H-%M-%S")}')
    os.makedirs(log_dir, exist_ok = True)
    s.get_data_setting().save_unloaded_wsis(log_dir)
    run(data, s, log_dir, runs_start=0,runs=s.get_network_setting().get_runs())


# @MPR
def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hd:c:w:q:", ["data_directory=","csv_file=","working_directory=", "qupath_project_file="])
    except getopt.GetoptError:
        print('main.py -d <data_directory> -c <csv_file> -w <working_directory> -q <qupath_project_file>')
        sys.exit(2)
    opts_vals = [o[0] for o in opts]
    if not('-d' in opts_vals or '--data_directory' in opts_vals):
        print('Specify -d or --data_directory')
        sys.exit(2)
    if not('-c' in opts_vals or '--csv_file' in opts_vals):
        print('Specify -c or --csv_file')
        sys.exit(2)
    if not('-w' in opts_vals or '--working_directory' in opts_vals):
        print('Specify -w or --working_directory')
        sys.exit(2)
    if not('-q' in opts_vals or '--qupath_project_file' in opts_vals):
        print('Specify -q or --qupath_project_file')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
             print('main.py -d <data_directory> -c <csv_file> -w <working_directory> -q <qupath_project_file>')
        elif opt in ('-d', '--data_directory'):
            data_directory = arg.strip('[]').split(',')
        elif opt in ('-c', '--csv_file'):
            if type(arg) == str and arg.endswith('.csv'):
                csv_file = arg
            else:
                print("Wrong data type for -c or --csv_file should be path to .csv")
                sys.exit(2)
        elif opt in ('-w', '--working_directiory'):
            if type(arg) == str:
                working_directory = arg
            else:
                print("Wrong data type for -w or --working_directory should be string")
                sys.exit(2)
        elif opt in ('-q', '--qupath_project:file'):
            if type(arg) == str and arg.endswith('.qpproj'):
                qupath_project_file = arg
            else:
                print("Wrong data type for -q or --qupath_project_file should be path to .qpproj")
                sys.exit(2)

    run_train(data_directory, csv_file, working_directory, qupath_project_file)

if __name__=="__main__":
    #run_train()
    main(sys.argv[1:])
