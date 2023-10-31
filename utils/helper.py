# @MPR
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def create_folder(folder):
        
    subfolders = folder.split('/')
    current_path = ''

    for sf in subfolders:
        current_path += sf + '/'

        if not os.path.exists(current_path):
            os.mkdir(current_path)

def plot_loss(log_dir, title):
    df = pd.read_csv(os.path.join(log_dir, "results.csv"))
    epochs = df["Epoch"]
    train_losses = df["Train Loss"]
    validation_losses = df["Val Loss"]
    dice_score = df["Val Dice Score"]
    
    fig, axes = plt.subplots(1, 2, figsize = (25, 7))
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].set_xticks(epochs)
    axes[0].plot(train_losses, label = "Train Loss")
    axes[0].plot(validation_losses, label = "Validation Loss")
    axes[0].legend()
    
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].grid(True)
    axes[1].set_xticks(epochs)
    axes[1].plot(dice_score, label = "Overall Val Dice Score", color = "red")
    axes[1].set_ylim((0, 1))
    axes[1].legend()
    fig.suptitle(title)
    fig.set_facecolor("white")
    fig.savefig(os.path.join(log_dir, "results_loss.png"))

def plot_dices(log_dir, title):
    df = pd.read_csv(os.path.join(log_dir, "results.csv"))
    epochs = df["Epoch"]
    dice_by_class_start_index =  df.columns.get_loc('Val Recall') + 1
    dice_scores = df["Val Dice Score"]

    fig, ax = plt.subplots(1, figsize = (20, 7))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice Score")
    ax.grid(True)
    ax.set_xticks(epochs)
    ax.plot(dice_scores, label = "Overall Val Dice Score", ls = "--", linewidth=3)
    for cls in df.columns[dice_by_class_start_index:]:
        ax.plot(df[cls], label = cls.split("_")[1] + "_DS")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.suptitle(title)
    fig.set_facecolor("white")
    fig.subplots_adjust(right=0.8)
    fig.savefig(os.path.join(log_dir, "results_ind_dice.png"))

def plot_accuracy(log_dir, title):
    df = pd.read_csv(os.path.join(log_dir, "results.csv"))
    epochs = df["Epoch"]
    train_accuracies = df["Train Accuracy"]
    train_foreground_accuracies = df["Train Foreground Accuracy"]
    val_accuracies = df["Val Accuracy"]
    val_foreground_accuracies = df["Val Foreground Accuracy"]
    
    fig, axes = plt.subplots(1, 2, figsize = (20, 7))
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True)
    axes[0].set_xticks(epochs)
    axes[0].plot(train_accuracies, label = "Train Accuracy")
    axes[0].plot(val_accuracies, label = "Validation Accuracy")
    axes[0].legend()
    axes[0].set_ylim((0, 1))
    
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Foreground Accuracy")
    axes[1].grid(True)
    axes[1].set_xticks(epochs)
    axes[1].plot(train_foreground_accuracies, label = "Train Accuracy")
    axes[1].plot(val_foreground_accuracies, label = "Validation Accuracy")
    axes[1].legend()
    axes[1].set_ylim((0, 1))
    fig.suptitle(title)
    fig.set_facecolor("white")
    fig.savefig(os.path.join(log_dir, "results_accuracy.png"))

def plot_examples(logdir, images, annotations, preds, identifier, n_classes):
    inference_folder = os.path.join(logdir, "inferences")
    create_folder(inference_folder)
    images_min, images_max = images.min(keepdims = True), images.max(keepdims = True)
    images = (images - images_min) / (images_max - images_min)
    for i in range(images.shape[0]):
        plt.imsave(os.path.join(inference_folder, f"{identifier}{i}_image.png"), 
                   np.transpose(images[i], [1, 2, 0]))
        # TODO: Update the following lines as if annotations have multiple channels for the future
        plt.imsave(os.path.join(inference_folder, f"{identifier}{i}_annotation.png"), 
                   annotations[i], vmin = 0, vmax = n_classes)
        plt.imsave(os.path.join(inference_folder, f"{identifier}{i}_pred.png"), 
                   preds[i], vmin = 0, vmax = n_classes)
        
def update_qupath_paths(qupath_project, file_folder_path, delete_not_found = False):
    """Use this function to replace old paths in a qupath project with the new ones

    Args:
        qupath_project mothi.tiling_projects.QuPathTilingProject: QuPathProject
        file_folder_path string: folder holding the wsis
        delete_not_found (bool, optional): Removes the wsis that not found Defaults to False.

    """
    uri2uri = {}
    images = os.listdir(file_folder_path)
    to_be_deleted = []
    for image in qupath_project.images:
        if image.image_name in images:
            old_path = image.uri
            new_path = "file:/" + os.path.join(file_folder_path, image.image_name).replace("\\", "/")
            uri2uri[old_path] = new_path
            print(f"Path of the {image.image_name} has been changed!")
        elif delete_not_found:
            to_be_deleted.append(image)
            print(f"{image.image_name} has been deleted!")
    qupath_project.update_image_paths(uri2uri = uri2uri)
    for image in to_be_deleted:
        qupath_project.remove_image(image)
    qupath_project.save()

def print_training_setting_summary(logdir, run, network_settings, train_set, validation_set,
                                    output_channels, device, targets, classes, annotation_threshold,):
    target_file = open(os.path.join(logdir, f"run_{run}_training_settings.txt"), "w")
    content = "TRAINING SETTINGS SUMMARY\n\nModel Parameters\n\n"
    content += f"Input_Channels: {network_settings.get_in_channels()}\n"
    content += f"Output Channels: {output_channels}\n"
    content += f"Target Indexes: {targets}\n"
    content += f"Target Classes: {classes}\n"
    content += f"Annotation Threshold: {annotation_threshold}\n"
    content += f"Loss Weights: {network_settings.get_loss_weights()}\n"
    content += f"Depth: {network_settings.get_depth()}\n"
    content += f"Wf: {network_settings.get_wf()}\n"
    content += f"Batch_Norm: {network_settings.get_batch_norm()}\n"
    content += f"Up_Mode: {network_settings.get_up_mode()}\n\n"
    content += "Training Parameters\n\n"
    content += f"Device: {device}\n"
    content += f"Runs: {network_settings.get_runs()}\n"
    content += f"Current_Run: {run}\n"
    content += f"Epochs: {network_settings.get_epochs()}\n"
    content += f"Batch_Size: {network_settings.get_batch_size()}\n"
    content += f"Overall Training Patches: {train_set.total_patches}\n"
    content += f"Training Patches After Filtering: {train_set.filtered_patches}\n"
    content += f"Training Patches Per Epoch: {network_settings.get_patches_per_epoch()}\n"
    content += f"Overall Validation Patches: {validation_set.total_patches}\n"
    content += f"Validation Patches After Filtering: {validation_set.filtered_patches}\n"
    content += f"Validation Patches Per Epoch: {network_settings.get_validation_patch_limit()}\n"
    print(content)
    print(content, file = target_file)

def print_testing_setting_summary(logdir, run, network_settings, test_set, 
                                  output_channels, device, targets, classes, annotation_threshold):
    target_file = open(os.path.join(logdir, f"run_{run}_testing_settings.txt"), "w")
    content = "TESTING SETTINGS SUMMARY\n\nModel Parameters\n\n"
    content += f"Input_Channels: {network_settings.get_in_channels()}\n"
    content += f"Output Channels: {output_channels}\n"
    content += f"Target Indexes: {targets}\n"
    content += f"Target Classes: {classes}\n"
    content += f"Annotation Threshold: {annotation_threshold}\n"
    content += f"Loss Weights: {network_settings.get_loss_weights()}\n"
    content += f"Depth: {network_settings.get_depth()}\n"
    content += f"Wf: {network_settings.get_wf()}\n"
    content += f"Batch_Norm: {network_settings.get_batch_norm()}\n"
    content += f"Up_Mode: {network_settings.get_up_mode()}\n\n"
    content += "Testing Parameters\n\n"
    content += f"Device: {device}\n"
    content += f"Runs: {network_settings.get_runs()}\n"
    content += f"Current_Run: {run}\n"
    content += f"Batch_Size: {network_settings.get_batch_size()}\n"
    content += f"Overall Testing Patches: {test_set.total_patches}\n"
    content += f"Testing Patches After Filtering: {test_set.filtered_patches}\n"
    content += f"Testing Patches Per Epoch: {network_settings.get_test_patch_limit()}\n"
    print(content)
    print(content, file = target_file)