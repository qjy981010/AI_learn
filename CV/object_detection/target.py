import torch
import random
import numpy as np

from utils import *

def rpn_target(anchors, gt_boxes, im_info, args):
    anchor_num = anchors.size(0)
    mask_inside = (
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] < im_info[1]) &
        (anchors[:, 3] < im_info[0]) &
    )
    anchors_inside = anchors[mask_inside]

    ious = calc_iou(anchors_inside, gt_boxes)
    anchor_max_iou, anchor_max_iou_idx = torch.max(ious, 1)  # len(anchors_inside)
    gtbox_max_iou_idx = torch.max(ious, 0)[1]  # len(gt_boxes)
    
    # labels:  1:positive   0:negative   -1:nothing
    labels = torch.zeros(len(anchors_inside)).fill_(-1)
    labels[anchor_max_iou < args.neg_iou_threshold] = 0
    labels[gtbox_max_iou_idx] = 1
    labels[anchor_max_iou >= args.pos_iou_threshold] = 1

    pos_idx = np.where(labels == 1)[0]
    pos_num = int(0.5 * arg.rpn_batch_size)
    if len(pos_idx) > pos_num:
        labels[random.sample(pos_idx, len(pos_idx)-pos_num)] = -1
    neg_idx = np.where(labels == 0)[0]
    neg_num = args.rpn_batch_size - min(len(pos_idx), pos_num)
    if len(neg_idx) > neg_num:
        labels[random.sample(neg_idx, len(neg_idx)-neg_num)] = -1

    offset = get_offset(anchors_inside, gt_boxes[anchor_max_iou_idx])
    weights = torch.zeros((len(offset), 4))
    pos_mask = labels == 1
    weights[pos_mask, :] = [1.0, 1.0, 1.0, 1.0]
    return labels, offset, weights


def out_target(proposals, gt_boxes, gt_labels, args):
    jittered_boxes = jitter_box(gt_boxes)
    all_boxes = torch.cat((proposals, jittered_boxes, gt_boxes), 0)
    
    ious = calc_iou(all_boxes, gt_boxes)
    boxes_max_iou, boxes_max_iou_idx = torch.max(ious, 1)  # len(all_boxes)
    gtbox_max_iou_idx = torch.max(ious, 0)[1]  # len(gt_boxes)
    labels = gt_labels[boxes_max_iou_idx]

    roi_num = args.out_batch_size
    fg_idx = np.where(boxes_max_iou >= args.fg_threshold)[0]
    fg_roi_num = int(roi_num * args.fg_fraction)
    if (len(fg_idx) > fg_roi_num):
        fg_idx = random.sample(fg_idx, fg_roi_num)
    fg_roi_num = min(fg_roi_num, len(fg_idx))

    bg_idx = np.where(0 <= boxes_max_iou < args.bg_threshold)[0]
    bg_roi_num = roi_num - fg_roi_num
    if (len(bg_idx) > bg_roi_num):
        bg_idx = random.sample(bg_idx, bg_roi_num)
    fbg_idx = np.append(fg_idx, bg_idx)

    labels = labels[fbg_idx]
    fbg_boxes = all_boxes[fbg_idx]
    labels[fg_roi_num:] = 0

    offset = get_offset(fbg_boxes, gt_boxes[boxes_max_iou_idx[fbg_idx], :])
    targets = torch.zeros((len(labels), 4 * 21))
    weights = torch.zeros(targets.size())
    for idx in range(fg_roi_num):
        start = 4 * int(labels[idx])
        end = start + 4
        targets[idx, start:end] = offset[idx, :]
        weights[idx, start:end] = [1, 1, 1, 1]

    return labels, fbg_boxes, targets, weights
