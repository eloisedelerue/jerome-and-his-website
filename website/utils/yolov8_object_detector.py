
# Libraries Import
import numpy as np
import torch
import torch.nn as nn
import cv2
import torchvision
from ultralytics import YOLO
from utils.deep_utils import Box

# YOLOv8 Object Detector Class
class YOLOv8ObjectDetector(nn.Module):
    def __init__(self, model_path, device, img_size=(640, 640), confidence=0.4, iou_thresh=0.45):
        """
        Constructor; loads the YOLOv8 model into memory.

        Args:
            model_path: Path to the YOLOv8 model file.
            device: Device to run the model on (e.g., 'cuda' or 'cpu').
            img_size: Input image size (height, width).
            confidence: Confidence threshold for detections.
            iou_thresh: IoU threshold for NMS.
        """
        super(YOLOv8ObjectDetector, self).__init__()
        self.device = device
        self.img_size = img_size
        self.confidence = confidence
        self.iou_thresh = iou_thresh
        print(f"[INFO] Loading YOLOv8 model from {model_path}...")
        self.yolo_wrapper = YOLO(model_path)
        self.model = self.yolo_wrapper.model
        self.model.requires_grad_(True)
        self.model.to(device)
        self.model.eval()
        self.names = self.model.names

    def preprocessing(self, img):
        """
        Converts a pixel matrix into a tensor.

        Args:
            img: Input image (numpy array).

        Returns:
            Tensor ready for model input, and original image shape.
        """
        original_shape = img.shape[:2]
        img_resized = cv2.resize(img, self.img_size)
        img_input = img_resized.transpose((2, 0, 1))
        img_input = np.ascontiguousarray(img_input)
        img_input = torch.from_numpy(img_input).to(self.device)
        img_input = img_input.float() / 255.0
        if len(img_input.shape) == 3:
            img_input = img_input.unsqueeze(0)
        return img_input, original_shape

    def custom_nms(self, preds):
        """
        Custom Non-Maximum Suppression using torchvision.

        Args:
            preds: Raw model predictions.

        Returns:
            List of filtered detections after NMS.
        """
        preds = preds.transpose(1, 2)
        output_list = []
        for i, pred in enumerate(preds):
            boxes = pred[:, :4]
            scores = pred[:, 4:]
            class_conf, class_pred = torch.max(scores, 1, keepdim=False)
            conf_mask = class_conf > self.confidence
            boxes = boxes[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if boxes.size(0) == 0:
                output_list.append([])
                continue
            box_xyxy = boxes.clone()
            box_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            box_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            box_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            box_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
            keep_indices = torchvision.ops.nms(box_xyxy, class_conf, self.iou_thresh)
            final_boxes = box_xyxy[keep_indices]
            final_scores = class_conf[keep_indices]
            final_classes = class_pred[keep_indices]
            out = torch.cat((final_boxes, final_scores.unsqueeze(1), final_classes.unsqueeze(1).float()), 1)
            output_list.append(out)
        return output_list

    def forward(self, img, original_shape=None):
        """
        Forward pass; returns detection boxes and raw predictions with gradients.

        Args:
            img: Input image tensor.
            original_shape: Original image shape for rescaling.

        Returns:
            List of boxes, classes, names, and confidences; and raw predictions.
        """
        preds = self.model(img)
        if isinstance(preds, list) or isinstance(preds, tuple):
            preds = preds[0]
        nms_preds = self.custom_nms(preds)

        boxes_list, classes_list, names_list, confs_list = [], [], [], []

        for i, det in enumerate(nms_preds):
            img_boxes, img_classes, img_names, img_confs = [], [], [], []

            if len(det) > 0:
                if original_shape is not None:
                    h, w = original_shape
                    scale_h, scale_w = h / self.img_size[1], w / self.img_size[0]
                    det[:, 0] *= scale_w; det[:, 1] *= scale_h
                    det[:, 2] *= scale_w; det[:, 3] *= scale_h
                    det[:, :4] = det[:, :4].round()

                for *xyxy, conf, cls in det:
                    bbox = Box.box2box([c.detach() for c in xyxy], Box.BoxSource.Torch, Box.BoxSource.Numpy, True)
                    img_boxes.append(bbox)
                    img_confs.append(round(conf.item(), 2))
                    cls_id = int(cls.item())
                    img_classes.append(cls_id)
                    img_names.append(self.names[cls_id])

            boxes_list.append(img_boxes)
            classes_list.append(img_classes)
            names_list.append(img_names)
            confs_list.append(img_confs)

        return [boxes_list, classes_list, names_list, confs_list], preds
