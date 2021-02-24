#!/usr/bin/env python3
"""Module for the class YOLO
"""

import tensorflow.keras as K
import numpy as np


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

    def process_outputs(self, outputs, image_size):
        """
        Args:
            outputs: list of numpy.ndarrays containing the predictions from
            the Darknet model for a single image.
            image_size: numpy.ndarray containing the image’s original
            size [image_height, image_width]

        Returns:
            A tuple of (boxes, box_confidences, box_class_probs)
        """

        boxes = []
        box_confidences = []
        box_class_probs = []
        height, width = image_size

        for idx, output in enumerate(outputs):
            (grid_h, grid_w, in_archons, _) = output.shape
            box = output[:, :, :, :4]
            for h in range(grid_h):
                for w in range(grid_w):
                    for a in range(in_archons):
                        pw, ph = self.anchors[idx][a]
                        tx, ty, tw, th = output[h, w, a, 0:4]
                        bx = (1 / (1 + np.exp(tx))) + w
                        bx = bx / grid_w
                        by = (1 / (1 + np.exp(ty))) + h
                        by = by / grid_h
                        bw = pw * np.exp(tw)
                        bw = bw / self.model.input.shape[1].value
                        bh = ph * np.exp(th)
                        bh = bh / self.model.input.shape[2].value
                        x1 = (bx - bw / 2)
                        x2 = (x1 + bw)
                        y1 = (by - bh / 2)
                        y2 = (y1 + bh)
                        box[h, w, a, 0] = x1 * width
                        box[h, w, a, 1] = y1 * height
                        box[h, w, a, 2] = x2 * width
                        box[h, w, a, 3] = y2 * height
            boxes.append(box)
            box_confidences.append(1 / (1 + np.exp(output[:, :, :, 4:5])))
            box_class_probs.append(1 / (1 + np.exp(output[:, :, :, 5:])))
        return boxes, box_confidences, box_class_probs
