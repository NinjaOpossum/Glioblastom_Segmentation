# @MPR 
from torch.utils.data import Dataset
from histomicstk.preprocessing.color_normalization import reinhard

import numpy as np

class PathologyDataset(Dataset):
    def __init__(self, dataset, infer = False, patch_size = 256, 
                 targets = [i for i in range(17)], annotation_threshold = None,
                 max_full_background_ratio = None):
        self.dataset = dataset
        self.wsi_tile_pairs = []
        self.patch_size = patch_size
        self.infer = infer
        # color norm. standard (from TCGA-A2-A3XS-DX1, Amgad et al, 2019)
        # self.cnorm = {
        #    'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
        #    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
        #}

        first_wsi = dataset[0][0]
        first_wsi.load_wsis()
        self.classes = [i.name for i in first_wsi.get_wsis()[0][0].get_qupath_project().path_classes]
        self.classes = [i for i in self.classes if i != "Unsicher"]
        self.total_patches = 0
        self.annotation_threshold = annotation_threshold
        self.targets = targets
        self.class_filter = [i - 1 for i in targets] + [len(self.classes)] # We add unsure class too to reject it
        self.full_background_tiles = 0
        self.max_full_background_ratio = max_full_background_ratio
        for c in dataset:
            # Iterate patients in class
            for p in c:
                p.load_wsis()
                # Iterate image properties
                for wp in p.get_wsis():
                    # Iterate WSIs with image property
                    for wsi in wp:
                        for tiles in wsi.tiles:
                            self.total_patches += len(tiles)
                            for tile in tiles:
                                if self.infer:
                                    self.wsi_tile_pairs.append([wsi, tile])
                                else:
                                    # Check if tile should be rejected
                                    result = tile.get_annotation(wsi, 
                                                                 annotation_threshold = self.annotation_threshold, 
                                                                 class_filter = self.class_filter)
                                    if result is None or result[1]:
                                        continue
                                    if max_full_background_ratio and (result[0] == 0).all():
                                        if self.full_background_tiles / (len(self.wsi_tile_pairs) + 1) > max_full_background_ratio:
                                            continue
                                        self.full_background_tiles += 1
                                    self.wsi_tile_pairs.append([wsi, tile])
        self.filtered_patches = len(self.wsi_tile_pairs)
        self.set_target_dicts()

    def __len__(self):
        return len(self.wsi_tile_pairs)

    def __getitem__(self, idx):
        wsi, tile = self.wsi_tile_pairs[idx]
        wsi.load_wsi()
        tissue = tile.get_image(wsi.image)
        # Do not bring the annotations if we are planning to do the inference
        if not self.infer:
            annotation, _ = tile.get_annotation(wsi, class_filter = self.class_filter)
            for qupath_index, model_index in self.forward_dict.items():
                annotation[annotation == qupath_index] = model_index
        # tissue_normalized = PathologyDataset.colormatch(tissue, self.cnorm)
        tissue_normalized = tissue
        tissue_transposed = np.transpose(self.pad(tissue_normalized), [2, 0, 1])
        if self.infer:
            return tissue_transposed.astype(np.float32), wsi.size, wsi.used_level, wsi.identifier, tile.position
        wsi.close_wsi()
        return tissue_transposed, annotation
    
    def pad(self, a):
        """Return bottom right padding."""
        zeros = np.zeros((self.patch_size, self.patch_size, a.shape[2]), dtype = np.float32)
        zeros[:a.shape[0], :a.shape[1]] = a
        return zeros


    # For possible color normalisation
    # def min_max_image(img):
    #     img_min = np.min(img, axis = (0, 1))
    #     img_max = np.max(img, axis = (0, 1))
    #     return (img - img_min) / (img_max - img_min)
    
    # def colormatch(img, cnorm = {
    #         'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
    #         'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
    #     }):
    #     # We can do the masked(masking non tissue parts) color normalization too
    #     # To test the masked normalization simply follow this page:
    #     # https://digitalslidearchive.github.io/HistomicsTK/examples/color_normalization_and_augmentation.html
    #     img_min_max = PathologyDataset.min_max_image(img)
    #     img_matched = reinhard(
    #         img_min_max, target_mu = cnorm['mu'], target_sigma = cnorm['sigma'])
    #     return img_matched / 255
    
    def set_target_dicts(self):
        # Dictionaries to hold sequential class indexes by the actual indexes(QupathId + 1)
        self.forward_dict = dict()
        self.backward_dict = dict()
        for i in range(len(self.targets)):
            if i != self.targets[i]:
                self.forward_dict[self.targets[i]] = i
                self.backward_dict[i] = self.targets[i]