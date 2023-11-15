# @MPR 
from torch.utils.data import Dataset
import cv2

import numpy as np

class PathologyDataset(Dataset):
    def __init__(self, dataset, infer = False, patch_size = 256, 
                 targets = [i for i in range(17)], annotation_threshold = None,
                 max_full_background_ratio = None,
                 do_fit_annotations = True,
                 air_label_num = 5):
        self.dataset = dataset
        self.wsi_tile_pairs = []
        self.patch_size = patch_size
        self.infer = infer

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
        self.do_fit_annotations = do_fit_annotations
        self.air_label_num = air_label_num

    def __len__(self):
        return len(self.wsi_tile_pairs)

    def __getitem__(self, idx):
        wsi, tile = self.wsi_tile_pairs[idx]
        wsi.load_wsi()
        tissue = tile.get_image(wsi.image)
        tissue = np.array(tissue)[:, :, :3]
        # Do not bring the annotations if we are planning to do the inference
        if not self.infer:
            annotation, _ = tile.get_annotation(wsi, class_filter = self.class_filter)
            if self.do_fit_annotations:
                if self.air_label_num in annotation and annotation.max() > 0:
                    otsu_air = self.get_treashold_air(tissue, annotation)
                else:
                    otsu_air = None
                # get ROI's for labels (0 background, 1 for label area)
                mask = self.fit_annotations(tissue, annotation, otsu_air)
                annotation = np.where(mask == 1, annotation, np.zeros(annotation.shape)).astype(np.uint8)
            for qupath_index, model_index in self.forward_dict.items():
                annotation[annotation == qupath_index] = model_index

        tissue_normalized = tissue / 255
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
    
    def set_target_dicts(self):
        # Dictionaries to hold sequential class indexes by the actual indexes(QupathId + 1)
        self.forward_dict = dict()
        self.backward_dict = dict()
        for i in range(len(self.targets)):
            if i != self.targets[i]:
                self.forward_dict[self.targets[i]] = i
                self.backward_dict[i] = self.targets[i]

    def fit_annotations(self, img, label, otsu = None):
        ''' clear the space between annotations / labels and image data

        Paramters:

            img: image to apply a better label fit
            label: label or annotations which should be fitted to the image

        Returns:
            fitted_mask_area: the mask (values between 0 and 1) for the fitted threshold areas
        '''
        min_contour_area_size = 100
        kernel_size = 5
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
        label = np.expand_dims(label, 2)
        img = np.where(label > 0, img, np.ones(np.shape(img))*255.).astype(np.uint8)

        imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        masked_img = img[img!=255]
        if otsu == None:
            otsu, _ =  cv2.threshold(masked_img, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        _, mask = cv2.threshold(imgg, otsu,255,cv2.THRESH_BINARY_INV)


        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, structuring_element, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, structuring_element, iterations=1)

        fitted_mask_area = np.zeros(label.squeeze().shape, dtype="uint8")


        # dismiss small or inner contour Areas
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > min_contour_area_size:
                fitted_mask_area = cv2.fillPoly(fitted_mask_area, [contour], 1)
        return fitted_mask_area
    
    def get_treashold_air(self, img, label):
        '''get the trashold value for air bubbles

        Paramters:

            img: image to extract otsu
            label: label image
            air_label: air annotations area

        Returns:
            otsu: treshholdvalue air annotation
        '''
        label = np.expand_dims(label, 2)
        img = np.where(label == self.air_label_num, img, np.ones(img.shape)*255).astype(np.uint16)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        otsu, _ =  cv2.threshold(img_grey, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        return otsu
