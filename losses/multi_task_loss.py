import torch
import torch.nn as nn

from losses.base_loss import F1Loss, FocalLoss, LabelSmoothingLoss, ArcFaceLoss, ClassBalancedLoss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function.
    Loss weights are learnt via homoscedastic uncertainty (Kendallet al.)
    """

    def __init__(self, losses_on, init_loss_weights=None, reduction="mean", eps=1e-6):
        """
        :param losses_on: List of outputs to apply losses on.
        Subset of ['FocalLoss', 'LabelSmoothingLoss', 'F1Loss', 'ClassBalancedLoss].
        :param init_loss_weights: Initial multi-task loss weights.
        :param reduction: 'mean' or 'sum'
        :param eps: small constant
        """
        super(MultiTaskLoss, self).__init__()

        self.losses_on = losses_on
        assert reduction in ["mean", "sum"], "Invalid reduction for loss."

        init_focal_log_var = 0
        init_label_smoothing_log_var = 0
        init_f1_log_var = 0
        init_class_balanced_log_var = 0
        init_arc_face_log_var = 0

        self.focal_log_var = nn.Parameter(torch.tensor(init_focal_log_var).float(), requires_grad=False)
        self.label_smoothing_log_var = nn.Parameter(
            torch.tensor(init_label_smoothing_log_var).float(), requires_grad=True
        )
        self.f1_log_var = nn.Parameter(torch.tensor(init_f1_log_var).float(), requires_grad=True)
        self.class_balanced_log_var = nn.Parameter(
            torch.tensor(init_class_balanced_log_var).float(), requires_grad=True
        )
        self.arc_face_log_var = nn.Parameter(torch.tensor(init_arc_face_log_var).float(), requires_grad=True)

        if "FocalLoss" in losses_on:
            self.focal_log_var.requires_grad = True
            self.focal_loss = FocalLoss()
        if "LabelSmoothingLoss" in losses_on:
            self.label_smoothing_log_var.requires_grad = True
            self.label_smoothing_loss = LabelSmoothingLoss(classes=18, smoothing=0.02)
        if "F1Loss" in losses_on:
            self.f1_log_var.requires_grad = True
            self.f1_loss = F1Loss(
                classes=18,
            )
        if "ClassBalancedLoss" in losses_on:
            self.class_balanced_log_var.requires_grad = True
            self.class_balanced_loss = ClassBalancedLoss()
        if "ArcFaceLoss" in losses_on:
            self.arc_face_log_var.requires.grad = True
            self.label_smoothing_loss = ArcFaceLoss()

    def forward(self, outputs, labels, is_train=True):
        total_loss = 0.0
        loss_dict = {}
        prefix = "train" if is_train else "val"

        if "FocalLoss" in self.losses_on:
            focal_loss = self.focal_loss(outputs, labels)
            total_loss += focal_loss * torch.exp(-self.focal_log_var) + self.focal_log_var
            loss_dict[prefix + "FocalLoss"] = focal_loss * torch.exp(-self.focal_log_var)
            loss_dict["focal_log_var"] = self.focal_log_var.data

        if "LabelSmoothingLoss" in self.losses_on:
            label_smoothing_loss = self.label_smoothing_loss(outputs, labels)
            total_loss += label_smoothing_loss * torch.exp(-self.label_smoothing_log_var) + self.label_smoothing_log_var
            loss_dict[prefix + "LabelSmoothingLoss"] = label_smoothing_loss * torch.exp(-self.label_smoothing_log_var)
            loss_dict["label_smoothing_log_var"] = self.label_smoothing_log_var.data

        if "F1Loss" in self.losses_on:
            f1_loss = self.f1_loss(outputs, labels)
            total_loss += f1_loss * torch.exp(-self.f1_log_var) + self.f1_log_var
            loss_dict[prefix + "F1Loss"] = f1_loss * torch.exp(-self.f1_log_var)
            loss_dict["f1_log_var"] = self.f1_log_var.data

        if "ClassBalancedLoss" in self.losses_on:
            class_balanced_loss = self.class_balanced_loss(outputs, labels)
            total_loss += class_balanced_loss * torch.exp(-self.class_balanced_log_var) + self.class_balanced_log_var
            loss_dict[prefix + "ClassBalancedLoss"] = class_balanced_loss * torch.exp(-self.class_balanced_log_var)
            loss_dict["class_balanced_log_var"] = self.class_balanced_log_var.data

        if "ArcFaceLoss" in self.losses_on:
            arc_face_loss = self.arc_face_loss(outputs, labels)
            total_loss += arc_face_loss * torch.exp(-self.arc_face_log_var) + self.arc_face_log_var
            loss_dict[prefix + "ArcFaceLoss"] = arc_face_loss * torch.exp(-self.arc_face_log_var)
            loss_dict["arc_face_log_var"] = self.arc_face_log_var.data

        return total_loss, loss_dict
