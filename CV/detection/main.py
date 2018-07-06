#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
from torchvision import transforms
from utils import *
from target import *

def train(net):
    # args.min_size = 
    # args.topn = 6000
    # args.post_topn = 2000
    # args.nms_threshold = 0.7
    # args.neg_iou_threshold = 0.3
    # args.pos_iou_threshold = 0.7
    # args.fg_threshold =
    # args.bg_threshold =
    # arg.rpn_batch_size = 256
    pass
    for image, gt_box, label, origin_shape in trainloader:
        # init
        h, w = image.size()[2:]
        origin_h, origin_w = origin_shape
        im_info = (h, w, origin_h, origin_w, h/origin_h, w/origin_w)
        gt_box[:, [0, 2]] *= im_info[5]
        gt_box[:, [1, 3]] *= im_info[4]

        # feature
        features = net.cnn(image)
        rpn_cls_prob, rpn_loc_offset = net.rpn(features)

        # region
        anchors = get_anchor(anchor_shape, *features.size()[-2:],
                             net.feat_stride)
        proposals, score = net.proposal_layer(rpn_cls_prob, rpn_loc_offset,
                                              im_info, args)

        # targets
        rpn_label, rpn_offset, rpn_weight = rpn_target(anchors, gt_box, im_info, args)
        frcnn_label, roi_boxes, frcnn_offset, frcnn_weight = frcnn_target(proposals, gt_box, args)
        if roi_boxes.size(0) == 0:
            continue

        # frcnn
        roi_features = net.roipool(features, roi_boxes)
        cls_result, loc_result = net.detect_net(roi_features)
