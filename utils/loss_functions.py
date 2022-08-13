import torch
import torch.nn as nn
from torch.nn import functional as F


def get_criterion(loss_type, negative_weight: float = 1, positive_weight: float = 1):

    if loss_type == 'RMSE':
        criterion = root_mean_square_error_loss
    elif loss_type == 'L2':
        criterion = torch.nn.MSELoss()
    elif loss_type == 'KLDivergence':
        criterion = torch.nn.KLDivLoss()
    elif loss_type == 'SmoothL1':
        criterion = torch.nn.SmoothL1Loss()
    else:
        raise Exception(f'unknown loss {loss_type}')

    return criterion


# define rmse loss with pytorch tensor operations
def root_mean_square_error_loss(logits: torch.Tensor, target: torch.Tensor):
    # logits shape B, C, H, W
    probs = torch.nn.functional.softmax(logits, dim=1)
    nominator = torch.sum(torch.pow(torch.sub(probs, target), 2))
    denominator = target.numel()
    output = torch.sqrt(nominator / denominator)
    return output


def soft_dice_loss(y_logit: torch.Tensor, y_true: torch.Tensor):
    y_prob = torch.sigmoid(y_logit)
    eps = 1e-6

    y_prob = y_prob.flatten()
    y_true = y_true.flatten()
    intersection = (y_prob * y_true).sum()

    return 1 - ((2. * intersection + eps) / (y_prob.sum() + y_true.sum() + eps))


# TODO: fix this one
def soft_dice_squared_sum_loss(y_logit: torch.Tensor, y_true: torch.Tensor):
    y_prob = torch.sigmoid(y_logit)
    eps = 1e-6

    y_prob = y_prob.flatten()
    y_true = y_true.flatten()
    intersection = (y_prob * y_true).sum()

    return 1 - ((2. * intersection + eps) / (y_prob.sum() + y_true.sum() + eps))


def soft_dice_loss_multi_class(input:torch.Tensor, y:torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    sum_dims= (0, 2, 3) # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom =  (y.sum(dim=sum_dims) + p.sum(dim=sum_dims)).clamp(eps)

    loss = 1 - (2. * intersection / denom).mean()
    return loss


def soft_dice_loss_multi_class_debug(input:torch.Tensor, y:torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    sum_dims= (0, 2, 3) # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom =  (y.sum(dim=sum_dims) + p.sum(dim=sum_dims)).clamp(eps)

    loss = 1 - (2. * intersection / denom).mean()
    loss_components = 1 - 2 * intersection/denom
    return loss, loss_components


def generalized_soft_dice_loss_multi_class(input:torch.Tensor, y:torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-12

    # TODO [B, C, H, W] -> [C, B, H, W] because softdice includes all pixels

    sum_dims= (0, 2, 3) # Batch, height, width
    ysum = y.sum(dim=sum_dims)
    wc = 1 / (ysum ** 2 + eps)
    intersection = ((y * p).sum(dim=sum_dims) * wc).sum()
    denom =  ((ysum + p.sum(dim=sum_dims)) * wc).sum()

    loss = 1 - (2. * intersection / denom)
    return loss


def jaccard_like_loss_multi_class(input:torch.Tensor, y:torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    # TODO [B, C, H, W] -> [C, B, H, W] because softdice includes all pixels

    sum_dims= (0, 2, 3) # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom =  (y ** 2 + p ** 2).sum(dim=sum_dims) + (y*p).sum(dim=sum_dims) + eps

    loss = 1 - (2. * intersection / denom).mean()
    return loss


def jaccard_like_loss(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() - (iflat * tflat).sum() + eps

    return 1 - ((2. * intersection) / denom)


def dice_like_loss(input: torch.Tensor, target: torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() + eps

    return 1 - ((2. * intersection) / denom)


def power_jaccard_loss(input: torch.Tensor, target: torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() - (iflat * tflat).sum() + eps

    return 1 - (intersection / denom)


def iou_loss(y_logit: torch.Tensor, y_true: torch.Tensor):
    y_pred = torch.sigmoid(y_logit)
    eps = 1e-6

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    intersection = (y_pred * y_true).sum()
    union = (y_pred + y_true).sum() - intersection + eps

    return 1 - (intersection / union)


def jaccard_like_balanced_loss(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() - (iflat * tflat).sum() + eps
    piccard = (2. * intersection)/denom

    n_iflat = 1-iflat
    n_tflat = 1-tflat
    neg_intersection = (n_iflat * n_tflat).sum()
    neg_denom = (n_iflat**2 + n_tflat**2).sum() - (n_iflat * n_tflat).sum()
    n_piccard = (2. * neg_intersection)/neg_denom

    return 1 - piccard - n_piccard


def soft_dice_loss_balanced(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    dice_pos = ((2. * intersection) /
                (iflat.sum() + tflat.sum() + eps))

    negatiev_intersection = ((1-iflat) * (1 - tflat)).sum()
    dice_neg =  (2 * negatiev_intersection) / ((1-iflat).sum() + (1-tflat).sum() + eps)

    return 1 - dice_pos - dice_neg


if __name__ == '__main__':
    import numpy as np
    criterion = torch.nn.MSELoss().to('cpu')
    pred = torch.tensor([-1000, 1, 2]).double()
    pred = torch.sigmoid(pred)
    print(pred)
    gt = torch.tensor([0, 1, 2]).double()
    mse = criterion(pred, gt)
    print(mse)
