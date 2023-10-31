# @MPR
from utils import helper
import torch

class NetworkSettingNew(object):

    def __init__(self, working_directory):
        """
        Parameters
        ----------
        working_directory : string
            Working directory for data
        """

        # Model Parameters
        self.in_channels = 3
        # n_classes will be retrieved from the class_setting
        self.depth = 5
        self.wf = 6
        self.batch_norm = True
        self.up_mode='upconv'

        # Training Parameters
        self.runs = 1
        self.epochs = 5
        self.patches_per_epoch = 10000
        self.validation_patch_limit = 1000
        self.test_patch_limit = 1000
        self.batch_size = 6
        self.early_stopping = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Stop the training if validation loss doesn't improve for 10 epochs(early_stopping should be True)
        self.patience = 10

        # Weights per classes, should be a tensor of size C (weight per channel)
        # Target indexes of the classes
        # 0 is background and foreground classes starts from the 1
        # 1: Tumor, 2: Hirnparenchym, 3: Infiltrationszone, 4: Blutung, 5: Luftblasen, 6: Haare
        # 7: Falten, 8: Risse, 9: Nekrosen, 10: Pseudopalisaden, 11: Artefakt, 12: artifiziell verändertes Gewebe
        # 13: Nekrose und Blutung, 14: Pränekrose, 15: out of focus, 16: Filzstift

        # For choosen classes
        # self.targets = [0, 1, 9, 12]

        # For all classes
        self.targets = [i for i in range(17)]

        # Skip tiles with less than %x threshold
        # Default value is None, you can set it to 0.2 to filter patches with less than %20 annotations
        # Warning: This option can severly reduce the patches with background classes
        # You may want to removing 0 from the targets to remove the background class at all
        self.annotation_threshold = None
        self.loss_weights = torch.ones(len(self.targets)).to(self.device)

        # Cross Entropy Weight
        self.ce_weight = 1
        # Soft Dice Loss Weight
        self.sd_weight = 0

        # tensor Background, Tumor, Nekrose, artifiziell verändertes Gewebe:
        # self.loss_weights = torch.tensor([
        # 1.024935179, 0.523821067, 2.465643787, 1.409033934]).to(self.device)

        # tensor weights all 17 classes:
        # self.loss_weights = torch.tensor([0.6080283917211534, 0.12327747711581505, 0, 0.8345036814960267, 4.842808183759067, 
        #                                 2727.1603690503694, 2632.230672909553, 26.93982846538412, 213624.7307010019, 0.5802141884837545,
        #                                 22.973871054760775, 16.92899201120585, 0.33159691202702346, 1.8196920892680644, 2.983346088627322,
        #                                 13.533519701845204, 2267.6850690919778]
        #                                 ).to(self.device)

        if 0 not in self.targets: # We will reject the background
            self.annotation_threshold = 1.0
        assert not(0 in self.targets and self.annotation_threshold == 1.0), "Remove index 0 to set annotation threshold 1.0"
        # Logging Options
        self.run_identifier = "Pathology"

    def get_in_channels(self):
        return self.in_channels

    def get_depth(self):
        return self.depth
    
    def get_wf(self):
        return self.wf
    
    def get_batch_norm(self):
        return self.batch_norm
    
    def get_up_mode(self):
        return self.up_mode
    
    def get_F(self):
        return 1024

        #Model hyper_parameters

        # helper.create_folder(self.model_folder)

    def get_runs(self):
        return self.runs
    
    def get_epochs(self):
        return self.epochs
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_patches_per_epoch(self):
        return self.patches_per_epoch
    
    def get_validation_patch_limit(self):
        return self.validation_patch_limit
    
    def get_test_patch_limit(self):
        return self.test_patch_limit
    
    def get_run_identifier(self):
        return self.run_identifier
    
    def get_loss_weights(self):
        return self.loss_weights
    
    def get_early_stopping(self):
        return self.early_stopping
    
    def get_patience(self):
        return self.patience
    
    def get_targets(self):
        return self.targets
    
    def get_annotation_threshold(self):
        return self.annotation_threshold
    
    def get_device(self):
        return self.device
    
    def get_ce_weight(self):
        return self.ce_weight
    
    def get_sd_weight(self):
        return self.sd_weight