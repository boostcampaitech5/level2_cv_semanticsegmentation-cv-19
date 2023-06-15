import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.lovasz_losses import LovaszLoss


# alpha=0.25 at notice board of aistages
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        BCE_EXP = torch.exp(-BCE)
        loss = self.alpha * (1 - BCE_EXP) ** self.gamma * BCE

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = 1 - ((2.0 * intersection + self.smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth))

        return loss.mean()


class IoULoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection + self.smooth) / (union + self.smooth)

        return 1 - IoU


# It deals with FP, FN
# When alpha=beta=0.5, it works as DiceLoss, (weight=4/3 can be used later)
# https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5
class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=0.7, beta=0.3):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)

        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=0.7, beta=0.3, gamma=1):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky


class LovaszHingeLoss(nn.Module):
    def __init__(self, per_image=False):
        super(LovaszHingeLoss, self).__init__()
        self.lovasz = LovaszLoss(mode="multilabel", per_image=per_image)

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        Lovasz = self.lovasz(inputs, targets)

        return Lovasz


# Hard-coded BCE + DiceLoss loss
class ComboLoss(nn.Module):
    def __init__(self, bce_weight=0.5) -> None:
        super(ComboLoss, self).__init__()
        self.bce_weight = bce_weight
        self.diceloss = DiceLoss()

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred = F.sigmoid(pred)
        dice = self.diceloss(pred, target)
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)

        return loss


_criterion_entrypoints = {
    "cross_entropy": nn.CrossEntropyLoss,
    "focal": FocalLoss,
    "bce_with_logit": nn.BCEWithLogitsLoss,
    "dice": DiceLoss,
    "iou": IoULoss,
    "combo_loss": ComboLoss,
    "tversky": TverskyLoss,
    "focal_tversky": FocalTverskyLoss,
    "lovaz": LovaszHingeLoss,
}


def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError("Unknown loss (%s)" % criterion_name)
    return criterion


class DiceCoef:
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):
        y_true_f = y_true.flatten(2)
        y_pred_f = y_pred.flatten(2)
        intersection = torch.sum(y_true_f * y_pred_f, -1)

        eps = 0.0001

        return (2.0 * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)


if __name__ == "__main__":
    criterion = []

    criterion_lst = ["tversky", "dice", "focal"]
    for i in criterion_lst:
        criterion.append(create_criterion(i))  # default: [bce_with_logit]
    print("criterion : ", criterion)

    import torch

    a = torch.rand([4, 29, 512, 512])
    b = torch.rand([4, 29, 512, 512])

    for i in criterion:
        loss = i(a, b)
        print(f"{i} : ", loss)
