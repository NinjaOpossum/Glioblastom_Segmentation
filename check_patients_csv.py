import os, sys
import random
import pandas as pd
import getopt

def fix_patients(file_path, patient_csv_path, fix_broken, new_file_name):
    patients = pd.read_csv(patient_csv_path)
    files = []
    for i in file_path:
        files += os.listdir(i)

    file_identifiers = set([f.split('_')[0] for f in files])
    problematic_csv_identifiers = []
    problematic_files = []
    for identifier in file_identifiers:
        print(identifier)
        if not patients['Histological ID'].eq(identifier).any():
            print(f"{identifier} is not exists in the patients")
            if fix_broken:
                arr = [len(patients), identifier, 'Meningioma benign', 1, 'Meningioma benign-1', 1, '-', -1, random.choice(['M', 'W'])]
                patients.loc[len(patients)] = arr
            problematic_csv_identifiers.append(identifier)
    if fix_broken:
        patients.to_csv(new_file_name, index = False)
    for i in files:
        if len(i.split("-")) == 3:
            print(f"{i} has a bugi name!")
            problematic_files.append(i)
    print("\n".join(problematic_files), file = open(new_file_name.split(".")[0] + "_problematic_files.txt", "w"))
    print("\n".join(problematic_csv_identifiers), file = open(new_file_name.split(".")[0] + "_problematic_csv_identifiers.txt", "w"))

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "d:c:f:n:", ["data_directory=","csv_file=","fix_broken=", "new_file_name="])
    except getopt.GetoptError:
        print('main.py -d <data_directory> -c <csv_file> -f <fix_broken> -n <new_file_name>')
        sys.exit(2)
    opts_vals = [o[0] for o in opts]
    if not('-d' in opts_vals or '--data_directory' in opts_vals):
        print('Specify -d or --data_directory')
        sys.exit(2)
    if not('-c' in opts_vals or '--csv_file' in opts_vals):
        print('Specify -c or --csv_file')
        sys.exit(2)
    if not('-f' in opts_vals or '--fix_broken' in opts_vals):
        print("fix_broken is set to True.")
        fix_broken = True
    if not('-n' in opts_vals or '--qupath_project_file' in opts_vals):
        print('New file name is set to  patients_fixed.csv')
        new_file_name = "patients_fixed.csv"

    for opt, arg in opts:
        if opt == '-h':
             print('main.py -d <data_directory> -c <csv_file> -f <fix_broken> -n <new_file_name>')
        elif opt in ('-d', '--data_directory'):
            data_directory = arg.strip('[]').split(',')
        elif opt in ('-c', '--csv_file'):
            if type(arg) == str and arg.endswith('.csv'):
                csv_file = arg
            else:
                print("Wrong data type for -c or --csv_file should be path to .csv")
                sys.exit(2)
        elif opt in ('-f', '--fix_broken'):
            if arg.lower() == "true" or arg.lower() == "false":
                fix_broken = arg.lower() == "true"
            else:
                print("Wrong value for -f or --fix_broken should be True of Flase")
                sys.exit(2)
        elif opt in ('-n', '--new_file_name'):
            if type(arg) == str and arg.endswith('.csv'):
                new_file_name = arg

    fix_patients(data_directory, csv_file, fix_broken, new_file_name)

if __name__=="__main__":
    #run_train()
    main(sys.argv[1:])