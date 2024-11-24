import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class Predictor:
    def __init__(self, model_cfg, checkpoint, device):
        self.device = device
        self.model = build_sam2(model_cfg, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(self.model)
        self.image_set = False

    def set_image(self, image):
        """Set the image for SAM prediction."""
        self.image = image
        self.predictor.set_image(image)
        self.image_set = True

    def predict(self, point_coords, point_labels, multimask_output=False):
        """Run SAM prediction."""
        if not self.image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        return self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output
        )
