#!/usr/bin/env python3
"""Module for the class YOLO
"""

import tensorflow.keras as K
import os
import cv2


class Yolo:
    """Uses Yolo V3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initializer for the Yolo class
        """

        self.model = K.models.load_model(model_path)
        self.class_names = []
        with open(classes_path, "r") as classes:
            for line in classes:
                self.class_names.append(line[:-1])
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def load_images(folder_path):
        """Loads images

        Args:
            folder_path: a string representing the path to the folder holding
            all the images to load

        Returns:
            A tuple of (images, image_paths)
        """

        load_image = K.preprocessing.image.load_img
        img_to_array = K.preprocessing.image.img_to_array
        images = []
        image_paths = []

        for img in os.listdir(folder_path):
            path = folder_path + "/" + img
            image = cv2.imread(path)
            if image is not None:
                images.append(image)
                image_paths.append(path)
        return images, image_paths
