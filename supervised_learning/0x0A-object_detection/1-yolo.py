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
            (grid_h, grid_w, n_archons, _) = output.shape
            box_confidences.append(1 / (1 + np.exp(-(output[:, :, :, 4:5]))))
            box_class_probs.append(1 / (1 + np.exp(-(output[:, :, :, 5:]))))
            box = output[:, :, :, :4]
            c = np.zeros((grid_h, grid_w, n_archons), dtype=int)

            idx_y = np.arange(grid_h)
            idx_y = idx_y.reshape(grid_h, 1, 1)
            idx_x = np.arange(grid_w)
            idx_x = idx_x.reshape(1, grid_w, 1)

            cy = c + idx_y
            cx = c + idx_x

            t_x = (box[..., 0])
            t_y = (box[..., 1])

            t_x_n = 1 / (1 + np.exp(-(t_x)))
            t_y_n = 1 / (1 + np.exp(-(t_y)))

            t_w = (box[..., 2])
            t_h = (box[..., 3])

            bx = t_x_n + cx
            by = t_y_n + cy

            bx /= grid_w
            by /= grid_h

            pw = self.anchors[idx, :, 0]
            ph = self.anchors[idx, :, 1]

            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)

            bw /= self.model.input.shape[1].value
            bh /= self.model.input.shape[2].value

            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh

            box[..., 0] = x1 * width
            box[..., 1] = y1 * height
            box[..., 2] = x2 * width
            box[..., 3] = y2 * height

            boxes.append(box)
        return boxes, box_confidences, box_class_probs
