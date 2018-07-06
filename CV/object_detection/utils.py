import torch

anchor_shape = torch.Tensor([
    [ -84,  -40,   99,   55],
    [-176,  -88,  191,  103],
    [-360, -184,  375,  199],
    [ -56,  -56,   71,   71],
    [-120, -120,  135,  135],
    [-248, -248,  263,  263],
    [ -36,  -80,   51,   95],
    [ -80, -168,   95,  183],
    [-168, -344,  183,  359],
])

def get_anchor(anchor_shape, height, width, feat_stride=16):
    grid_x = torch.arange(0, width) * feat_stride
    grid_y = torch.arange(0, height) * feat_stride

    grid_x = grid_x.view(-1, 1).repeat(1, height).view(height * width, 1)
    grid_y = grid_y.repeat(width, 1).view(height * width, 1)

    grid = torch.cat((grid_x, grid_y, grid_x, grid_y), 1)

    anchors = anchor_shape.unsqueeze(0) + grid.unsqueeze(1)

    return anchors.view(-1, 4)

def calc_iou(boxes, gt_boxes):
    x1, y1, x2, y2 = boxes.permute(1, 0)
    boxes_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    result = []
    for box in gt_boxes:
        gt_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

        in_x1 = torch.clamp(x1, min=box[0])
        in_y1 = torch.clamp(y1, min=box[1])
        in_x2 = torch.clamp(x2, min=box[2])
        in_y2 = torch.clamp(y2, min=box[3])

        width = torch.clamp(in_x2 - in_x1 + 1, min=0.0)
        height = torch.clamp(in_y2 - in_y1 + 1, min=0.0)
        in_area = width * height
        iou = torch.div(in_area, (gt_area + boxes_area - in_area))
        
        result.append(iou.view(-1, 1))
    return torch.cat(result, 1)  # len(boxes) x len(gt_boxes)

def get_offset(boxes, gt_boxes):
    width = boxes[:, 2] - boxes[:, 0] + 1.0
    height = boxes[:, 3] - boxes[:, 1] + 1.0
    center_x = boxes[:, 0] + 0.5 * width
    center_y = boxes[:, 1] + 0.5 * height

    gt_width = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_height = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_center_x = gt_boxes[:, 0] + 0.5 * gt_width
    gt_center_y = gt_boxes[:, 1] + 0.5 * gt_height

    dw = torch.log(gt_width / width)
    dh = torch.log(gt_height / height)
    dx = (gt_center_x - center_x) / width
    dy = (gt_center_y - center_y) / height
    return torch.cat((dx, dy, dw, dh), 0).t()

def jitter_box(boxes, jitter_rate=0.05):
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    x_offset = (np.random.rand(len(boxes)) - 0.5) * jitter_rate * width
    y_offset = (np.random.rand(len(boxes)) - 0.5) * jitter_rate * height
    jittered_boxes = boxes.copy()
    jittered_boxes[:, [0, 2]] += x_offset
    jittered_boxes[:, [1, 3]] += y_offset
    return torch.clamp(jittered_boxes, min=0.0)
