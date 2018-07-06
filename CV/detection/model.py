import torch
import math
import torch.nn as nn
from torchvision import models
from utils import anchor_shape, get_anchor

class ConvNet(nn.Module):
    """
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        original_model = models.vgg16(pretrained=True)
        # print(original_model)
        # print(list(original_model.features.children())[:-1])
        self.net = nn.Sequential(
            *list(original_model.features.children())[:-1])

    def forward(self, x):
        return self.net(x)


class RPN(nn.Module):
    """
    """

    def __init__(self):
        super(RPN, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1),
                                  nn.ReLU())
        self.cls_conv = nn.Conv2d(512, 9*2, 1, 1)  # 2: is foreground or not
        self.loc_conv = nn.Conv2d(512, 9*4, 1, 1)  # 4: len([x, y, h, w])
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        conv_result = self.conv(x)
        cls_result = self.cls_conv(conv_result)
        loc_result = self.loc_conv(conv_result)
        height, width = cls_result.size()[-2:]
        cls_result = cls_result.permute(0, 2, 3, 1).view(-1, 2)
        # cls_result = self.softmax(cls_result)
        # cls_result = cls_result.view(1, height, width, channel_num)
        # cls_result = cls_result.permute(0, 3, 1, 2)
        return cls_result, loc_result


class ProposalLayer(nn.Module):
    """
    """

    def __init__(self):
        super(ProposalLayer, self).__init__()
        
    def forward(self, cls_prob, loc_offset, im_info, args):
        anchors = get_anchor(anchor_shape, *loc_offset.size()[-2:])
        loc_offset = loc_offset.permute(0, 2, 3, 1).view(-1, 4)
        proposal = self._get_proposal(anchors, loc_offset)
        
        # clip proposals to image
        proposal = self._clip_proposal(proposal, *im_info[:2])

        # class score for every proposal
        score = cls_prob[:, 0].view(-1)

        # remove small proposals
        mask = self._del_small_idx(proposal, args.min_size * max(im_info[-2:]))
        proposal = proposal[mask]
        score = score[mask]

        # get proposals with top score
        top_idx = torch.sort(score.squeeze(), dim=0)[1][::-1][:args.topn]
        proposal, score = proposal[top_idx], score[top_idx]
        mask = self._nms(proposal, score, args.nms_threshold)[:args.post_topn]

        return proposal[mask], score[mask]

    def _nms(self, proposal, score, threshold):
        x1, y1, x2, y2 = proposal.permute(1, 0)
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idx_list = torch.arange(len(proposal))
        result = []
        while len(idx_list) > 0:
            i = idx_list[0]
            result.append(i)
            in_x1 = torch.clamp(x1[idx_list[1:]], min=x1[i])
            in_y1 = torch.clamp(y1[idx_list[1:]], min=y1[i])
            in_x2 = torch.clamp(x2[idx_list[1:]], max=x2[i])
            in_y2 = torch.clamp(y2[idx_list[1:]], max=y2[i])

            width = torch.clamp(in_x2 - in_x1 + 1, min=0.0)
            height = torch.clamp(in_y2 - in_y1 + 1, min=0.0)

            in_area = width * height
            iou = in_area / (area[i] + area[idx_list[1:]] - in_area)

            idx_list = idx_list[1:][iou <= threshold]
        return result

    def _del_small_idx(self, proposal, min_size):
        width = proposal[:, 2] - proposal[:, 0] + 1
        height = proposal[:, 3] - proposal[:, 1] + 1
        return (width > min_size) & (height > min_size)

    def _clip_proposal(self, proposal, img_height, img_width):
        proposal[:, 0] = torch.clamp(proposal[:, 0], 0, img_width-1)
        proposal[:, 1] = torch.clamp(proposal[:, 1], 0, img_height-1)
        proposal[:, 2] = torch.clamp(proposal[:, 2], 0, img_width-1)
        proposal[:, 3] = torch.clamp(proposal[:, 3], 0, img_height-1)
        return proposal

    def _get_proposal(self, anchors, loc_offset):
        width = anchors[:, 2] - anchors[:, 0] + 1.0
        height = anchors[:, 3] - anchors[:, 1] + 1.0
        center_x = anchors[:, 0] + 0.5 * width
        center_y = anchors[:, 1] + 0.5 * height

        dx, dy, dw, dh = loc_offset.permute(1, 0)

        width, height = width.unsqueeze(1), height.unsqueeze(1)
        center_x = dx * width + center_x.unsqueeze(1)
        center_y = dy * height + center_y.unsqueeze(1)
        width = torch.exp(dw) * width
        height = torch.exp(dh) * height

        proposal = torch.zeros(loc_offset.size())
        proposal[:, 0] = center_x - 0.5 * width
        proposal[:, 1] = center_y - 0.5 * height
        proposal[:, 2] = center_x + 0.5 * width
        proposal[:, 3] = center_y + 0.5 * height
        return proposal


class RoiPooling(nn.Module):
    """
    """
    
    def __init__(self, size=7, scale=1/16):
        super(RoiPooling, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d(size)
        self.scale = scale

    def forward(self, feature, proposal):
        proposal = torch.mul(proposal, self.scale).long()
        result = []
        for roi in proposal:
            region_feature = feature[:, :, roi[1]:roi[3]+1, roi[0]:roi[2]+1]
            pool_feature = self.pool(region_feature)
            result.append(pool_feature)
        return torch.cat(result, 0)


class DetectNet(nn.Module):
    """
    """
    
    def __init__(self):
        super(DetectNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(512*7*7, 4096),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(4096, 4096),
                                nn.ReLU(),
                                nn.Dropout())
        self.classifer = nn.Linear(4096, 21)
        self.softmax = nn.Softmax()
        self.regressor = nn.Linear(4096, 21*4)

    def forward(self, x):
        x = x.view(-1, 512*7*7)
        x = self.fc(x)
        cls_result = self.softmax(self.classifer(x))
        loc_result = self.regressor(x)
        return cls_result, loc_result
            

class FasterRCNN(nn.Module):
    """
    """
    
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.cnn = ConvNet()
        self.feat_stride = 16
        self.rpn = RPN()
        self.proposal_layer = ProposalLayer()
        self.roipool = RoiPooling(size=7, scale=1/self.feat_stride)
        self.detect_net = DetectNet()
        