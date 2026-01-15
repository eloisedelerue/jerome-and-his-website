
# Libraries Import
import torch
from pytorch_grad_cam import GradCAM

# YOLO Target Class
class YOLOTarget:
    """
    Isolates only the scores of the class of interest,
    sums all its scores over the image to create a single scalar score
    that Grad-CAM will seek to explain.
    """
    def __init__(self, category_index):
        self.category = category_index

    def __call__(self, model_output):
        if len(model_output.shape) == 3:
            return model_output[:, 4 + self.category, :].sum()
        return model_output.sum()

# YOLOV8 Grad-CAM Class
class YOLOV8GradCAM:
    """
    Acts as an intermediary between the YOLO detector and pytorch-grad-cam.
    """
    def __init__(self, model, layer_name, img_size=(640, 640)):
        self.model = model
        self.img_size = img_size
        target_layers = [self.model.model.model[9]] # Last layer of the backbone
        self.cam = GradCAM(model=self.model.model, target_layers=target_layers)

    def __call__(self, input_tensor, targets=None, original_shape=None):
        # FIX: Pass original_shape to the model!
        out, preds = self.model(input_tensor, original_shape)
        boxes, classes, names, confidences = out

        masks = []
        if len(boxes[0]) == 0:
            return [], None, out # If nothing is found

        for i, class_id in enumerate(classes[0]): # Focus on one class at a time
            targets = [YOLOTarget(class_id)]
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
            masks.append(grayscale_cam[0, :])

        return masks, None, out
