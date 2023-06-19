import torch
import torch.nn as nn
from model.base_model import BaseModel
from transformers import Mask2FormerForUniversalSegmentation


class Mask2Former_Swin_S(BaseModel):
    def __init__(self, num_classes=29):
        super(Mask2Former_Swin_S, self).__init__()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-small-ade-semantic", num_labels=num_classes, ignore_mismatched_sizes=True
        )

    def forward(self, image):
        outputs = self.model(pixel_values=image)

        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        masks_queries_logits_expanded = masks_queries_logits.unsqueeze(2)
        class_queries_logits_expanded = class_queries_logits.unsqueeze(3).unsqueeze(4)
        outputs = torch.sum(masks_queries_logits_expanded * class_queries_logits_expanded, dim=1)[:, 1:, ...]

        upsampled_logits = nn.functional.interpolate(outputs, size=image.shape[-1], mode="bilinear", align_corners=False)

        return upsampled_logits
