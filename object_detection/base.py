import torch
from matplotlib import pyplot as plt
from typing import List, Any


def _box_corner_to_center(boxes):
    """
    转换标记锚框的方法: corner -> center
    ARGS:
        - boxes: 使用corner方法标记的一组锚框
    RETURN:
        1. 使用center方法标记的一组锚框
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], axis=-1)


def _box_center_to_corner(boxes):
    """
    转换标记锚框的方法: center -> corner
    ARGS:
        - boxes: 使用center方法标记的一组锚框
    RETURN:
        1. 使用corner方法标记的一组锚框
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return torch.stack((x1, y1, x2, y2), axis=-1)


def _box_iou(boxes1, boxes2):
        """
        计算两组锚框(真实边界框)之间的iou
        ARGS:
            - boxes1: 第一组锚框(真实边界框), 2d <- (第一组锚框(真实边界框)数量, 4)
            - boxes2: 第二组锚框(真实边界框), 2d <- (第二组锚框(真实边界框)数量, 4)
        RETURN:
            1. 第一组的每一个锚框(真实边界框)分别与第二组的锚框(真实边界框)之间的交并比, 2d <- (boxes1.shape[0], boxes2.shape[0])
        """
        # 获得两组锚框(真实边界框)内所有box的面积
        # box_area通过 (x2-x1)*(y2-y1) 计算得到区域面积, 1d <- torch.Size([当前组锚框(真实边界框)数量])
        box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        areas1 = box_area(boxes1)
        areas2 = box_area(boxes2)

        # 计算交集的左上和右下坐标, 需将第一组的所有锚框(真实边界框)分别与第二组的每一个锚框(真实边界框)进行计算
        # inter_upperlefts是交集左上角坐标, 3d <- (boxes1.shape[0], boxes2.shape[0], 2)
        # inter_lowerrights是交集右下角坐标, 3d <- (boxes1.shape[0], boxes2.shape[0], 2)
        inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

        # 计算交集和并集面积
        # inter为所有交集的高宽, 如果值为负则置0, 3d <- (boxes1.shape[0], boxes2.shape[0], 2)
        inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
        inters_areas = inters[:, :, 0] * inters[:, :, 1]  # 交集面积, 2d <- (boxes1.shape[0], boxes2.shape[0])
        union_areas = areas1[:, None] + areas2 - inters_areas  # 并集面积, 2d <- (boxes1.shape[0], boxes2.shape[0])
        return inters_areas / union_areas


def multibox_prior(data, sizes: List[Any], ratios: List[Any]):
    """
    生成以每个像素为中心具有不同形状的锚框
    ARGS:
        - data: 特征图, 4d <- (BatchSize, Channel, Height, Width)
        - sizes: 锚框尺度
        - ratios: 锚框比例
    RETURN:
        1. 每个像素生成的所有锚框, 3d <- (1, boxes_per_pixel*特征图的像素个数, 4)
    """
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = num_sizes + num_ratios - 1
    sizes_tensor = torch.tensor(sizes, device=device)
    ratios_tensor = torch.tensor(ratios, device=device)

    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y轴上缩放步长
    steps_w = 1.0 / in_width  # 在x轴上缩放步长

    # 生成锚框的所有中心点
    # 最终shift_y shape: [center_h1, center_h1, center_h2, center_h2]
    # 最终shift_x: [center_w1, center_w2, center_w1, center_w2]
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 锚框宽度: s*sqrt(r)
    # 锚框高度: s/sqrt(r)
    # 's' 和 'r' 具体组合方式 (共 boxes_per_pixel个组合): {s1, r1}, {s1, r2}, ..., {s1, rn}, {s2, r1}, {s3, r1}, ..., {sn, r1}
    w = (
        torch.cat((sizes_tensor * torch.sqrt(ratios_tensor[0]), sizes_tensor[0] * torch.sqrt(ratios_tensor[1:])))
        * in_height
        / in_width
    )  # 's' 和 'r' 组合后计算得到的 boxes_per_pixel个宽, 1d
    h = torch.cat((sizes_tensor / torch.sqrt(ratios_tensor[0]), sizes_tensor[0] / torch.sqrt(ratios_tensor[1:])))
    anchor_manipulations = (
        torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
    )  # 锚框的一半高宽, 用于计算每个锚框的坐标, 2d <- (boxes_per_pixel*特征图像素个数, 4)

    # 正式生成锚框
    # 每个像素生成锚框时依照顺序: (center_w1, center_h1), (center_w2, center_h1), ..., (center_wn, center_h1), (center_w1, center_h2), ..., (center_wn, center_hn)
    # 最终返回在每个像素生成的所有锚框, 3d <- (1, boxes_per_pixel*特征图的像素个数, 4)
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)  #
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


def multibox_target(anchors, labels):
    """
    为锚框标记真实边界框和类别(不是每一个锚框都会匹配到真实边界框, 但所有真实边界框一定会匹配到锚框)
    ARGS:
        - anchors: 锚框, 3d <- (1, 锚框总数, 4)
        - labels: 真实边界框, 3d <- (BatchSize, 图片内真实边框数, 5)
    RETURN:
        1. 锚框对真实边界框的偏移, 2d <- (BatchSize, 锚框总数 * 4)
        2. 若锚框匹配到真实边界框则为1否则为0, 2d <- (BatchSize, 锚框总数 * 4)
        3. 每个锚框匹配到的类别, 2d <- (BatchSize, 锚框总数)
    """
    def _assign_anchor_to_bbox(anchors, ground_truth, device, iou_threshold=0.5):
        """
        将最接近的真实边界框分配给锚框
        ARGS:
            - anchors: 锚框, 2d <- (锚框总数, 4)
            - ground_truth: 真实边界框, 2d <- (真实边界框总数, 4)
            - device: 设备, cpu or cuda:x
            - iou_threshold: iou阈值, 默认为0.5
        RETURNS:
            1. 每个锚框匹配到的真实边界框的索引, -1表示未匹配到真实边界框(背景), 1d <- (anchors.shape[0])
        """
        num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]

        # jaccard为每个锚框分别与所有真实边界框的iou, 2d <- (anchors.shape[0], ground_truth.shape[0])
        jaccard = _box_iou(anchors, ground_truth)

        # 根据阈值，决定是否分配真实边界框
        # max_iou是针对每一个锚框，与其重叠度最高的真实边界框的iou值, 1d <- (anchors.shape[0])
        # indices是针对每一个锚框，与其重叠度最高的真实边框的索引, 1d <- (anchors.shape[0])
        # 最终assigned_map存储的是每个锚框匹配到的真实边界框的索引, 若为-1则标识该锚框未匹配到真实编辑框(作为背景), 1d <- (anchors.shape[0])
        assigned_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
        max_ious, indices = torch.max(jaccard, dim=1)
        # anchors_idx: 锚框与真实边界框匹配后大于阈值的锚框索引, 1d <- (大于等于阈值的iou个数)
        # bbox_idx: 锚框与真实边界框匹配后大于阈值的真实边界框的索引, 1d <- (大于等于阈值的iou个数)
        anchors_idx = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
        bboxes_idx = indices[max_ious >= iou_threshold]
        assigned_map[anchors_idx] = bboxes_idx

        # 修正assigned_map, 寻找每个真实边界框iou值最高的锚框, 并覆盖该锚框对应真实边界框的索引
        col_discard = torch.full((num_anchors,), -1)
        row_discard = torch.full((num_gt_boxes,), -1)
        for _ in range(num_gt_boxes):
            max_jaccard_idx = torch.argmax(jaccard)
            box_idx = (max_jaccard_idx % num_gt_boxes).long()
            anchors_idx = (max_jaccard_idx / num_gt_boxes).long()
            assigned_map[anchors_idx] = box_idx
            jaccard[:, box_idx] = -1
            jaccard[anchors_idx, :] = -1
        return assigned_map

    def _offset_boxes(anchors, assigned_bboxes, eps=1e-6):
        """
        计算每个锚框与其对应真实框之间的偏移量
        ARGS:
            - anchors: 锚框, 2d <- torch.Size([锚框总数, 4])
            - assigned_bboxes: 与锚框一一对应的真实框, 2d <- torch.Size([锚框总数, 4])
            - eps: 小的偏移, 防止被除数为0, 默认为1e-6
        RETURN
            1. 每个锚框与其对应真实框之间的偏移量, 2d <- torch.Size([锚框总数, 4])
        """
        anchors = _box_corner_to_center(anchors)  # 将锚框从corner标记转换为center标记, torch.Size([锚框总数, 4])
        assigned_bboxes = _box_corner_to_center(assigned_bboxes)
        offset_xy = (assigned_bboxes[:, :2] - anchors[:, :2]) / anchors[:, 2:] * 10  # torch.Size([锚框总数, 2])
        offset_wh = torch.log(eps + assigned_bboxes[:, 2:] / anchors[:, 2:]) * 5
        return torch.cat([offset_xy, offset_wh], axis=-1)

    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_cls_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]  # 当前图片的真实边框, 2d <- (当前图片内真实边框数, 5)
        box_map = _assign_anchor_to_bbox(anchors, label[:, 1:], device)
        # mask: 如果锚框并未匹配到真实边界框, 则mask中对应位置为0, 2d <- (anchors.shape[0], 4)
        mask = ((box_map >= 0).float().unsqueeze(-1)).repeat(1, 4)

        # cls_labels: 每一个锚框对应的真实边界框的类别id, 1d <- torch.Size([anchors.shape[0]])
        # assigned_bboxes: 每一个锚框对应的真实边界框的坐标, 2d <- torch.Size([anchors.shape[0], 4])
        # 注意: 若该锚框未匹配到真实边界框则其类别为背景(id=0), 若匹配到了真实边界框则其类别为对应真实边界框类别id+1
        cls_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bboxes = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        indices_true = torch.nonzero(box_map >= 0)  # 非背景锚框的索引, 2d <- (非背景锚框个数, 1)
        bboxes_idx = box_map[indices_true]  # 非背景锚框对应类被的索引, 2d <- (非背景锚框个数, 1)
        cls_labels[indices_true] = label[bboxes_idx, 0].long() + 1
        assigned_bboxes[indices_true] = label[bboxes_idx, 1:]

        # 计算偏移量
        # offset: 每个锚框与其真实框之间的偏移量, 如果该锚框为背景则偏移量为0, 2d <- torch.Size([anchors.shape[0], 4])
        offset = _offset_boxes(anchors, assigned_bboxes) * mask
        
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(mask.reshape(-1))
        batch_cls_labels.append(cls_labels)
    offset = torch.stack(batch_offset)    # 2d <- torch.Size([BatchSize, anchors.shape[0] * 4])
    mask = torch.stack(batch_mask)  # 2d <- torch.Size([BatchSize, anchors.shape[0] * 4])
    cls_labels = torch.stack(batch_cls_labels)  # 2d <- torch.Size([BatchSize, anchors.shape[0]])
    return (offset, mask, cls_labels)


def multibox_detection(anchors, cls_probs, offset_preds, nms_threshold=0.5, pos_threshold=0.009999999):
    """
        
    ARGS:
        - anchors: 锚框, 3d <- torch.Size([1, 锚框数, 4])
        - cls_probs: 小批量内每个类别下的所有锚框对该类别的置信度, 3d <- torch.Size([BatchSize, 类别数, 锚框数])
        - offset_preds: 小批量内每个锚框预测偏移组成的向量, 2d <- torch.Size([BatcSize, 锚框数*4])
        - nms_threshold: 用于nms计算的阈值, 默认为0.5
        - pos_threshold: 置信度低于该阈值的锚框作为背景
    RETURN:
        1. 所有图片的预测类别, 预测类别的置信度和预测出真实边框的位置, 3d <- torch.Size([BatchSize, 锚框数, 6])
    """

    def _offset_inverse(anchors, offset_preds):
        """
        使用锚框和偏移量找到真实框
        ARGS:
            - anchors: 锚框, 2d <- torch.Size([锚框总数, 4])
            - offset: 与锚框一一对应的偏移, 2d <- torch.Size([锚框总数, 4])
        RETURN:
            1. 每个锚框与其偏移量共同计算得到的真实框, 2d <- torch.Size([锚框总数, 4])
        """
        center_anchors = _box_corner_to_center(anchors)
        bbox_xy = (offset_preds[:, :2] * 0.1 * center_anchors[:, 2:]) + center_anchors[:, :2]
        bbox_wh = torch.exp(offset_preds[:, :2] * 0.2) * center_anchors[:, 2:]
        bbox = torch.cat([bbox_xy, bbox_wh], axis=1)
        return _box_center_to_corner(bbox)

    def _nms(boxes, scores, iou_threshold):
        """
        对预测边界框的置信度进行排序
        ARGS:
            - boxes: 边框, 2d <- torch.Size([边框数量, 4])
            - scores: 类别置信度, 1d <- torch.Size([边框数量])
            - iou_threshold: 阈值, 重叠度高于阈值且置信度低的边框将被遗弃
        RETURN:
            1. 与其他边框重叠度低于iou_threshold且置信度更高的边框(结果依据置信度从大到小排序存放), 1d <- torch.Size([剩余边框数])
        """
        B = torch.argsort(scores, dim=-1, descending=True)  # 将置信度从高到低排序后的索引, 1d <- torch.Size([boxes.shape[0]])
        keep = []
        while B.numel() > 0:
            i = B[0]
            keep.append(i)
            if B.numel() == 1:
                break
            # 当前最大置信度边框与其余边框的iou, 1d <- torch.Size([boxes.shape[0]-1])
            iou = _box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
            inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
            B = B[inds+1]
        return torch.tensor(keep, device=boxes.device)

    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_cls, num_anchor = cls_probs.shape[1], anchors.shape[0]
    out = []

    for i in range(batch_size):
        # offset_pred: reshape之后表示一个BatchSize下每个锚框预测的offset值, 2d <- torch.Size([anchors.shape[0], 4])
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)

        # 每个锚框的最大置信值及其所属类别(除背景外)
        # conf: 每个锚框最大置信值, 1d <- torch.Size([anchors.shape[0]])
        # class_id: 锚框最大置信值所属类别, 1d <- torch.Size([anchors.shape[0]])
        # keep: nms后剩余边框的索引,依据置信度排序, 1d <- torch.Size([剩余边框数])
        conf, class_id = torch.max(cls_prob[1:], dim=0)
        predicted_bb = _offset_inverse(anchors, offset_pred)  # 预测的真实框, 2d <- torch.Size([anchors.shape[0], 4])
        keep = _nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        # uniques: 全部索引按照从小到大排序, 1d <- torch.Size([anchors.shape[0]])
        # counts: 每一个索引在combined出现的次数, 1d <- torch.Size([anchors.shape[0]])
        all_idx = torch.arange(num_anchor, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))  # 合并keep和all_idx, 1d <- torch.Size([剩余边框数 + anchors.shape[0]])
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]  # 没有被保存在keep下的边框的索引

        # 新的排序方式, 优先排nms后保留的锚框(置信度从高到低), 后依顺序排其余锚框
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]

        # pred_info 表示当前批量下每个锚框的预测结果, 2d <- torch.Size([anchors.shape[0], 6])
        below_min_idx = conf < pos_threshold
        class_id[below_min_idx] = -1  # 置信度小于below_min_idx的锚框类别调整为-1(视作背景)
        conf[below_min_idx] = 1 - conf[below_min_idx]    # 当真实类别置信度均很低时, 背景置信度会高
        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """在图片中标记并展示"""

    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    def _bbox_to_rect(bbox, color):
        return plt.Rectangle(
            xy=(bbox[0], bbox[1]),
            width=bbox[2] - bbox[0],
            height=bbox[3] - bbox[1],
            fill=False,
            edgecolor=color,
            linewidth=2,
        )

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])

    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = _bbox_to_rect(bbox.numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(
                rect.xy[0],
                rect.xy[1],
                labels[i],
                va='center',
                ha='center',
                fontsize=9,
                color=text_color,
                bbox=dict(facecolor=color, lw=0),
            )
