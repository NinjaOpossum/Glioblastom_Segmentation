# utils.py
# @MPR 
import numpy as np
import torch


def dice_coef(y_true, y_pred, smooth=1e-6):
    '''
    Calculate and return the Dice coefficient for single label predictions.
    
    y_true : array
        Ground truth array (binary: 0 or 1).
    
    y_pred : array
        Predicted labels (binary: 0 or 1).
    
    smooth : float, optional (default=1e-6)
        A small constant added to avoid division by zero.
        
    Returns
    -------
    float
        Dice coefficient
    '''
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    y_sum = np.sum(y_true_f) + np.sum(y_pred_f)
    return (2 * intersection + smooth) / (y_sum + smooth)


def dice_coef_multilabel(y_true, y_pred, numLabels):
    '''
    Compute the Dice coefficient for multi-label predictions by averaging 
    the Dice coefficients calculated for each label.
    
    y_true : array
        Ground truth array with multiple labels.
    
    y_pred : array
        Predicted labels array with multiple labels.
        
    numLabels : int
        Number of possible labels.
        
    Returns
    -------
    float
        Average Dice coefficient across all labels.
    '''
    dice = 0
    for i in range(1, numLabels):
        y_true_norm = np.where(y_true == i, 1, 0).astype(np.uint8)
        y_pred_norm = np.where(y_pred == i, 1, 0).astype(np.uint8)
        dice += dice_coef(y_true_norm, y_pred_norm)
        
    return dice / numLabels

def dice_coef_per_label(y_true, y_pred, numLabels, start_index = 1):
    '''
    Calculate the Dice coefficient for each label in multi-label predictions.
    
    y_true : array
        Ground truth array with multiple labels.
    
    y_pred : array
        Predicted labels array with multiple labels.
        
    numLabels : int
        Number of possible labels.
        
    start_index : int, optional (default=1)
        Label index to start from.
        
    Returns
    -------
    dict
        Dictionary containing Dice coefficients for each label.
    '''
    label_dice_dict = {}
    for i in range(start_index, numLabels):
        y_true_norm = np.where(y_true == i, 1, 0).astype(np.uint8)
        y_pred_norm = np.where(y_pred == i, 1, 0).astype(np.uint8)
        label_dice_dict[i] = dice_coef(y_true_norm, y_pred_norm)
    return label_dice_dict

def get_precision_recall(pred_index, annotations, n_classes, start_index = 1):
    '''
    Calculate the mean precision and recall.
    
    pred_index : array
        Predicted labels.
    
    annotations : array
        Ground truth labels.
        
    n_classes : int
        Number of classes/labels.
        
    start_index : int, optional (default=1)
        Class index to start from.
        
    Returns
    -------
    tuple (float, float)
        Mean precision and recall.
    '''
    precision = 0
    recall = 0
    for i in range(start_index, n_classes):
        pred_index_class_bin = (pred_index == i).astype(int) 
        annotations_class_bin = (annotations == i).astype(int)
        tp = ((pred_index_class_bin == 1) * pred_index_class_bin == annotations_class_bin).sum()
        fp = ((pred_index_class_bin == 1) * pred_index_class_bin != annotations_class_bin).sum()
        fn = ((pred_index_class_bin == 0) * pred_index_class_bin != annotations_class_bin).sum()
        precision += tp / (tp + fp)
        recall += tp / (tp + fn)
    mean_precision = precision / (n_classes - 1)
    mean_recall = recall / (n_classes - 1)
    return mean_precision, mean_recall

class CompositeLoss(torch.nn.Module):
    '''
    A composite loss function that combines Cross Entropy Loss and 
    Soft Dice Loss. Allows for weighting of the different loss components.
    
    Attributes
    ----------
    ce : torch.nn.CrossEntropyLoss
        Cross-Entropy loss component.
    
    ce_weight : float
        Weight of the Cross-Entropy loss component.
        
    sd_weight : float
        Weight of the Soft Dice loss component.
    
    num_labels : int
        Number of possible labels.
        
    start_index : int
        Index to start from for loss calculation.
        
    loss_weights : array-like
        Weights assigned to each class/label.
    '''

    def __init__(self, num_labels, start_index, loss_weights, ce_weight = 1.0, sd_weight = 1.0):
        '''
        Initializes the CompositeLoss with given attributes.
        '''
        super(CompositeLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight = loss_weights)
        self.ce_weight = ce_weight
        self.sd_weight = sd_weight
        self.num_labels = num_labels
        self.start_index = start_index
        self.loss_weights = loss_weights

    def multi_class_sof_dice_loss(self, pred, annotations, smooth=1e-6):
        '''
        Calculate the multi-class soft dice loss.
        
        pred : array
            Predicted labels.
        
        annotations : array
            Ground truth labels.
            
        smooth : float, optional (default=1e-6)
            A small constant added to avoid division by zero.
            
        Returns
        -------
        float
            Soft Dice Loss.
        '''
        soft_dice = 0
        for i in range(self.start_index, self.num_labels):
            targets_mask = annotations == i
            probs = torch.softmax(pred, dim = 1)[:, i]
            cross = targets_mask * probs
            soft_dice += (2 * torch.sum(cross) + smooth) / (torch.sum(targets_mask) + torch.sum(probs) + smooth)\
                * self.loss_weights[i]
        return 1 - soft_dice / self.loss_weights[self.start_index:self.num_labels].sum()

    def forward(self, pred, annotations):
        '''
        Calculate the total loss as a weighted combination of Cross Entropy and Soft Dice Loss.
        
        pred : array
            Predicted labels.
        
        annotations : array
            Ground truth labels.
            
        Returns
        -------
        float
            Total loss.
        '''
        ce_loss = self.ce(pred, annotations)
        if self.sd_weight > 0:
            soft_dice_loss = self.multi_class_sof_dice_loss(pred, annotations)
            return (self.ce_weight * ce_loss  + soft_dice_loss * self.sd_weight) / (self.ce_weight + self.sd_weight)
        return ce_loss
