#!/usr/bin/env python3
"""Module for the class YOLO
"""

import cv2
import numpy as np
import os
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ Process the inputs to return the boxes with the max scores along
        with the scores and the class belonging to the box.

        Args:
            boxes: Contain the processed boundary boxes for each output
            box_confidences: Contain the processed box confidences for each
            output
            box_class_probs: Contain the processed box class probabilities for
            each output, respectively
        Returns:
            A tuple of (filtered_boxes, box_classes, box_scores)
        """

        filtered_boxes = []
        box_classes = []
        box_scores = []
        for idx, box in enumerate(boxes):
            scores = box_confidences[idx] * box_class_probs[idx]
            classes = np.argmax(scores, axis=-1)
            scores = np.max(scores, axis=-1)
            mask = scores >= self.class_t

            filtered_boxes += boxes[idx][mask].tolist()
            box_scores += scores[mask].tolist()
            box_classes += classes[mask].tolist()
        filtered_boxes = np.array(filtered_boxes)
        box_classes = np.array(box_classes)
        box_scores = np.array(box_scores, dtype=np.float32)

        return filtered_boxes, box_classes, box_scores

    @staticmethod
    def calc_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
        h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
        if w_intersection <= 0 or h_intersection <= 0:
            return 0
        i = w_intersection * h_intersection
        u = w1 * h1 + w2 * h2 - i
        return i / u

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Uses non-max suppresion to remove boxes with score less than nms_t
        and sorts them by class then by score

        Args:
            filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
            the filtered bounding boxes:
            box_classes: a numpy.ndarray of shape (?,) containing the class
            number for the class that filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?) containing the box scores
            for each box in filtered_boxes, respectively
        Returns:
            A tuple of (box_predictions, predicted_box_classes,
            predicted_box_scores)
        """

        if len(filtered_boxes) == 0:
            return []

        i_sorted = np.lexsort((-box_scores, box_classes))
        picks = []
        sorted_len = len(i_sorted)

        for idx in range(sorted_len - 1):
            i = idx + 1
            suppress = []
            if i < len(i_sorted):
                while (box_classes[i_sorted[idx]] ==
                       box_classes[i_sorted[i]]):
                    iou = self.calc_iou(filtered_boxes[i_sorted[idx]],
                                        filtered_boxes[i_sorted[i]])
                    if iou > self.nms_t:
                        suppress.append(i)
                    i += 1
                    if i >= len(i_sorted):
                        break
                idx = i
            i_sorted = np.delete(i_sorted, suppress)
        return (filtered_boxes[i_sorted],
                box_classes[i_sorted],
                box_scores[i_sorted])

    @staticmethod
    def load_images(folder_path):
        """Loads images

        Args:
            folder_path: a string representing the path to the folder holding
            all the images to load

        Returns:
            A tuple of (images, image_paths)
        """
        images = []
        image_paths = []

        for img in os.listdir(folder_path):
            path = folder_path + "/" + img
            image = cv2.imread(path)
            if image is not None:
                images.append(image)
                image_paths.append(path)
        return images, image_paths

    def preprocess_images(self, images):
        """ process images to the shape(416, 416, 3) using cubic interpolation
        then mormalizes the values of the channels

        Args:
            images: List of images as numpy.ndarrays
        Returns:
            A tuple of (pimages, image_shapes)
        """

        width = self.model.input_shape[1]
        height = self.model.input_shape[2]

        pimages = []
        image_shapes = []

        for image in images:
            image_height, image_width, _ = image.shape
            img = cv2.resize(image,
                             dsize=(width, height),
                             interpolation=cv2.INTER_CUBIC)
            img = img / 255
            pimages.append(img)
            image_shapes.append((image_height, image_width))
        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Displays the image with all boundary boxes, class names,
        and box scores

        Args:
            image: a numpy.ndarray containing an unprocessed image
            boxes: a numpy.ndarray containing the boundary boxes for the image
            box_classes: a numpy.ndarray containing the class indices for
            each box
            box_scores: a numpy.ndarray containing the box scores for each box
            file_name: the file path where the original image is stored
        """

        blue = (255, 0, 0)
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            image = np.array(image)
            image = cv2.rectangle(image,
                                  (x1, y1),
                                  (x2, y2),
                                  color=blue,
                                  thickness=2)
            box_text = (self.class_names[box_classes[idx]]
                        + " " + str(round(box_scores[idx], 2)))
            image = cv2.putText(image,
                                box_text,
                                (x1, y1 - 5),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                color=(0, 0, 255),
                                thickness=1,
                                lineType=cv2.LINE_AA)
        cv2.imshow(file_name, image)
        k = cv2.waitKey(0)
        if k == ord('s'):
            try:
                os.mkdir('detections')
            except FileExistsError:
                pass
            cv2.imwrite("detections/" + file_name, image)
            cv2.destroyAllWindows()
        else:
            cv2.destroyAllWindows()

    def predict(self, folder_path):
        """Makes predictions using the darknet model, then shows the images
        with the predictions.

        Args:
            folder_path: a string representing the path to the folder holding
            all the images to predict.
        Returns:
            A tuple of (predictions, image_paths)
        """

        p_outputs = []
        f_boxes = []
        fb_classes = []
        fb_scores = []
        f_pred = []

        images, img_paths = self.load_images(folder_path)
        pimages, img_shapes = self.preprocess_images(images)
        predictions = self.model.predict(pimages)

        for idx in range(len(img_paths)):
            outputs = []
            outputs.append(predictions[0][idx, :, :, :, :])
            outputs.append(predictions[1][idx, :, :, :, :])
            outputs.append(predictions[2][idx, :, :, :, :])
            (boxes,
             box_confidences,
             box_class_probs) = self.process_outputs(outputs, img_shapes[idx])
            boxes, box_classes, box_scores = self.filter_boxes(boxes,
                                                               box_confidences,
                                                               box_class_probs)

            for lst, ret in zip([f_boxes, fb_classes, fb_scores],
                                self.non_max_suppression(boxes,
                                                         box_classes,
                                                         box_scores)):
                lst.append(ret)

        for idx, image in enumerate(images):
            img_name = img_paths[idx].split("/")[-1]
            f_pred.append((f_boxes[idx], fb_classes[idx], fb_scores[idx]))
            self.show_boxes(image,
                            f_boxes[idx],
                            fb_classes[idx],
                            fb_scores[idx],
                            img_name)

        return f_pred, img_paths
