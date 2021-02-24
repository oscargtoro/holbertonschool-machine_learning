#!/usr/bin/env python3
"""Module for the class YOLO
"""

import tensorflow.keras as K


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
        for output in outputs:
            (grid_h, grid_w, anchors, _) = output.shape
            confidence = output[:, :, :, 0:1]
            box = output[:, :, :, 0:4]
            classes = output[:, :, :, 5:]
            for h in range(grid_h):
                for w in range(grid_w):
                    x1 = y1 = x2 = y2 = 0
                    confidences = []
                    for a in range(anchors):
                        x1 = output[h, w, a, 0] - (grid_w / 2)
                        y1 = output[h, w, a, 1] + (grid_h / 2)
                        x2 = output[h, w, a, 0] + (grid_w / 2)
                        y2 = output[h, w, a, 1] - (grid_h / 2)
                        box[h, w, a, :] = [x1, y1, x2, y2]
                        confidence[h, w, a, :] = output[h, w, a, 4]
                        classes[h, w, a, :] = output[h, w, a, 5:]
            boxes.append(box)
            box_confidences.append(confidence)
            box_class_probs.append(classes)
        return boxes, box_confidences, box_class_probs
