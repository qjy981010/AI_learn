import torch

def rpn_loss(cls_prob, loc_offset, labels, offset, weight):
    height, width = loc_offset.size()[-2:]
    idx = labels.ge(0).nonzero()[:, 0]
    cls_prob = cls_prob[idx, :]
    labels = labels[idx, :]