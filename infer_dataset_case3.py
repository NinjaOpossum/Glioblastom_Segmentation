import os, time
#os.add_dll_directory(r"C:\Users\PC\Desktop\Abrek\openslide-win64-20230414\openslide-win64-20230414\bin")

import arguments.setting as setting

# from learning.construct_features import construct_features
from progress.bar import IncrementalBar
from models import unet
from datastructure import dataset_torch
import datastructure.dataset as dataset

from torch.utils.data import DataLoader

# import learning.train
# import learning.test
import getopt, sys
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_infer(ann_mask, qupath_project, save_path, targets):
    classes = ["Background", "Tumor", "Residual"] #Ignore unsure class
    my_cmap = [(1., 1., 1.),(1., 0.0, 0.0), (147 / 255, 245 / 255, 66 / 255)]
    cmap = mpl.colors.ListedColormap(my_cmap)
    ratio_x = ann_mask.shape[0] / (ann_mask.shape[0] + ann_mask.shape[1])
    ratio_y = ann_mask.shape[1] / (ann_mask.shape[0] + ann_mask.shape[1])
    plt.figure(figsize=(int(32 * ratio_x * 2), int(32 * ratio_y * 2)), dpi=320)
    cax = plt.imshow(ann_mask.T[::50, ::50], cmap = cmap, vmin = 0, vmax = len(classes) - 1)
    cbar = plt.colorbar(cax, ticks = range(0, len(my_cmap)))
    cbar.ax.set_yticklabels(classes)  # vertically oriented colorbar
    plt.axis('off')
    plt.savefig(save_path, facecolor = "white")

@torch.no_grad()
def infer(dataloader, model, log_dir, device, patch_size, qupath_project, targets, backward_dict = None):
    ann_mask = None
    for tiles, size, level, identifier, position in dataloader:
        if ann_mask is None:
            ann_mask = np.zeros((size[0][0], size[1][0]), dtype = np.uint8)
            current_level = level[0]
            current_identifier = identifier[0]
            current_size = (size[0][0], size[1][0])
            print(f"Infering {current_identifier}!")
        tiles = tiles.to(device)
        pred = model(tiles).detach().cpu().numpy()
        # pred_map = pred.argmax(axis = 1).astype(np.uint8)
        pred_map = pred.sum(axis = (2, 3)).argmax(axis = 1).astype(np.uint8)
        for i in range(len(tiles)):
            if identifier[i] != current_identifier:
                print(f"Plotting {current_identifier}")
                plot_infer(ann_mask, qupath_project, os.path.join(log_dir, f"{current_identifier}_infer.png"), targets)
                print(f"Saving {current_identifier}")
                if backward_dict:
                    for model_index, qupath_index in reversed(backward_dict.items()):
                        ann_mask[ann_mask == model_index] = qupath_index
                np.save(os.path.join(log_dir, f"{current_identifier}_{current_level}_case1.npy"), ann_mask)
                current_level = level[i]
                current_identifier = identifier[i]
                print(f"Infering {current_identifier}!")
                current_size = (size[0][i], size[1][i])
                ann_mask = np.zeros((size[0][i], size[1][i]), dtype = np.uint8)
            pos_x = position[0][i]
            width = min(patch_size, current_size[0] - pos_x)
            pos_y = position[1][i]
            height = min(patch_size, current_size[1] - pos_y)
            # ann_mask[pos_x : pos_x + width, pos_y : pos_y + height] = np.transpose(pred_map[i], [1, 0])[:width, :height]
            ann_mask[pos_x : pos_x + width, pos_y : pos_y + height] = pred_map[i]
    print(f"Saving {current_identifier}!")
    np.save(os.path.join(log_dir, f"{current_identifier}_{current_level}.npy"), ann_mask)
    print(f"Plotting {current_identifier}")
    plot_infer(ann_mask, qupath_project, os.path.join(log_dir, f"{current_identifier}_infer.png"))

def run_infer(data_directories, csv_file, working_directory, qupath_project_file, kth_run, infer_set, model_path):
    """ Set up setting and dataset and run training/testing
    """
    s = setting.Setting(data_directories, csv_file, working_directory, qupath_project_file)
    network_settings = s.get_network_setting()
    data_settings = s.get_data_setting()

    data = dataset.Dataset(s)
    data.set_fold(kth_run)
    log_dir = os.path.join(working_directory, 'Runs', f'{s.get_network_setting().get_run_identifier()}_Infer-{time.strftime("%y-%m-%d-%H-%M-%S")}')
    os.makedirs(log_dir, exist_ok = True)
    if infer_set == "train":
        infer_dataset = data.get_train_set()
    elif infer_set == "validation":
        infer_dataset = data.get_validation_set()
    else:
        infer_dataset = data.get_test_set()

    targets = network_settings.get_targets()
    annotation_threshold = network_settings.get_annotation_threshold()
    use_background = int(annotation_threshold != 1.0)
    infer_dataset = dataset_torch.PathologyDataset(infer_dataset, True, data_settings.get_patch_size(),
                                                   targets, annotation_threshold)

    all_classes = infer_dataset.classes
    if not use_background:
        # There is no background class
        classes = [all_classes[i - 1] for i in targets]
    else:
        # There is a background class
        classes = [all_classes[i - 1] for i in targets[1:]]
    n_classes = 3
    
    batch_size = network_settings.get_batch_size()
    patch_size = s.get_data_setting().get_patch_size()
    infer_dataset = torch.utils.data.DataLoader(infer_dataset, pin_memory = True, batch_size = batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    model = unet.UNet(network_settings.get_in_channels(),
                n_classes,
                network_settings.get_depth(),
                network_settings.get_wf(),
                True,
                network_settings.get_batch_norm(),
                network_settings.get_up_mode())

    model.to(device)
    model.load_state_dict(torch.load(model_path))

    infer(infer_dataset, model, log_dir, device, patch_size, data_settings.get_qupath_project(), targets)



def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hd:c:w:q:k:s:m:", ["data_directory=","csv_file=","working_directory=", 
                                                          "qupath_project_file=", "k_run=", "set=",
                                                          "model_weights="])
    except getopt.GetoptError:
        print('main.py -d <data_directory> -c <csv_file> -w <working_directory> -q <qupath_project_file> -k <kth run> -s set_name -m model_weight_path')
        sys.exit(2)
    opts_vals = [o[0] for o in opts]
    print(opts)
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
    if not('-k' in opts_vals or '--k_run' in opts_vals):
        print('Specify -k or --k_run Example --k 0')
        sys.exit(2)
    if not('-s' in opts_vals or '--set' in opts_vals):
        print('Specify -s or --set Example --set test')
        sys.exit(2)
    if not('-m' in opts_vals or '--model_weights' in opts_vals):
        print('Specify -m or --model_weights')
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
        elif opt in ('-k', '--k_run'):
            try:
                kth_run = int(arg)
            except:
                print("Wrong data type or value for -k or --k_run should be integer")
                sys.exit(2)
        elif opt in ('-s', '--set'):
            if type(arg) == str:
                if arg.lower() in ["train", "test", "validation"]:
                    infer_set = arg.lower()
                else:
                    print("Wrong set name for -s or --set can be train | validation | test")
                    sys.exit(2)
            else:
                print("Wrong data type for -s or --set should be string")
                sys.exit(2)
        elif opt in ('-m', '--model_weights'):
            if type(arg) == str:
                model_path = arg
            else:
                print("Wrong data type name for -m or --model_weights should be string")


    run_infer(data_directory, csv_file, working_directory, qupath_project_file, kth_run, infer_set, model_path)

if __name__=="__main__":
    #run_train()
    main(sys.argv[1:])

