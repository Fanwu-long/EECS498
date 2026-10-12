import math
from typing import Dict, List, Optional

import torch
from a4_helper import *
from common import DetectorBackboneWithFPN, class_spec_nms, get_fpn_location_coords
from torch import nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision.ops import sigmoid_focal_loss

TensorDict = Dict[str, torch.Tensor]

def hello_one_stage_detector():
    print("Hello from one_stage_detector.py!")

class FCOSPredictionNetwork(nn.Module):
    """
    FCOS prediction network that accepts FPN feature maps from different levels
    and makes three predictions at every location: bounding boxes, class ID and
    centerness. This module contains a "stem" of convolution layers, along with
    one final layer per prediction. For a visual depiction, see Figure 2 (right
    side) in FCOS paper: https://arxiv.org/abs/1904.01355

    We will use feature maps from FPN levels (P3, P4, P5) and exclude (P6, P7).
    """

    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):
        """
        Args:
            num_classes: Number of object classes for classification.
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN, since the head directly
                operates on them.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
        """
        super().__init__()

        # cls / box 两条 stem，结构一样但参数不共享
        stem_cls = []
        stem_box = []
        prev = in_channels
        for ch in stem_channels:
            c1 = nn.Conv2d(prev, ch, kernel_size=3, stride=1, padding=1)
            nn.init.normal_(c1.weight, mean=0, std=0.01)
            nn.init.constant_(c1.bias, 0)
            stem_cls += [c1, nn.ReLU()]

            c2 = nn.Conv2d(prev, ch, kernel_size=3, stride=1, padding=1)
            nn.init.normal_(c2.weight, mean=0, std=0.01)
            nn.init.constant_(c2.bias, 0)
            stem_box += [c2, nn.ReLU()]
            prev = ch

        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)

        # 三个头：分类 / 框 / centerness
        last_ch = stem_channels[-1]
        self.pred_cls = nn.Conv2d(last_ch, num_classes, 3, padding=1)
        self.pred_box = nn.Conv2d(last_ch, 4, 3, padding=1)
        self.pred_ctr = nn.Conv2d(last_ch, 1, 3, padding=1)
        for layer in [self.pred_cls, self.pred_box, self.pred_ctr]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            nn.init.constant_(layer.bias, 0)

        torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict the desired outputs at every location
        (as described above). Format them such that channels are placed at the
        last dimension, and (H, W) are flattened (having channels at last is
        convenient for computing loss as well as perforning inference).

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}. Each
                tensor will have shape `(batch_size, fpn_channels, H, W)`. For an
                input (224, 224) image, H = W are (28, 14, 7) for (p3, p4, p5).

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Classification logits: `(batch_size, H * W, num_classes)`.
            2. Box regression deltas: `(batch_size, H * W, 4)`
            3. Centerness logits:     `(batch_size, H * W, 1)`
        """

        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}

        for level_name, feat in feats_per_fpn_level.items():
            B = feat.shape[0]
            cls_feat = self.stem_cls(feat)
            box_feat = self.stem_box(feat)
            cls = self.pred_cls(cls_feat)
            box = self.pred_box(box_feat)
            ctr = self.pred_ctr(box_feat)
            # 通道挪到最后，再把 H*W 摊平，方便算 loss
            class_logits[level_name] = cls.permute(0, 2, 3, 1).reshape(B, -1, cls.shape[1])
            boxreg_deltas[level_name] = box.permute(0, 2, 3, 1).reshape(B, -1, 4)
            centerness_logits[level_name] = ctr.permute(0, 2, 3, 1).reshape(B, -1, 1)

        return [class_logits, boxreg_deltas, centerness_logits]

@torch.no_grad()
# 每个location找最小面积且包含它的gt
def fcos_match_locations_to_gt(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    gt_boxes: torch.Tensor,
) -> TensorDict:
    """
    Match centers of the locations of FPN feature with a set of GT bounding
    boxes of the input image. Since our model makes predictions at every FPN
    feature map location, we must supervise it with an appropriate GT box.
    There are multiple GT boxes in image, so FCOS has a set of heuristics to
    assign centers with GT, which we implement here.

    NOTE: This function is NOT BATCHED. Call separately for GT box batches.

    Args:
        locations_per_fpn_level: Centers at different levels of FPN (p3, p4, p5),
            that are already projected to absolute co-ordinates in input image
            dimension. Dictionary of three keys: (p3, p4, p5) giving tensors of
            shape `(H * W, 2)` where H = W is the size of feature map.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `common.py` for more details.
        gt_boxes: GT boxes of a single image, a batch of `(M, 5)` boxes with
            absolute co-ordinates and class ID `(x1, y1, x2, y2, C)`. In this
            codebase, this tensor is directly served by the dataloader.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(N, 5)` GT boxes, one for each center. They are
            one of M input boxes, or a dummy box called "background" that is
            `(-1, -1, -1, -1, -1)`. Background indicates that the center does
            not belong to any object.
    """

    matched_gt_boxes = {
        level_name: None for level_name in locations_per_fpn_level.keys()
    }

    for level_name, centers in locations_per_fpn_level.items():

        stride = strides_per_fpn_level[level_name]

        x, y = centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes[:, :4].unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)

        pairwise_dist = pairwise_dist.permute(1, 0, 2)

        match_matrix = pairwise_dist.min(dim=2).values > 0

        pairwise_dist = pairwise_dist.max(dim=2).values

        lower_bound = stride * 4 if level_name != "p3" else 0
        upper_bound = stride * 8 if level_name != "p5" else float("inf")
        match_matrix &= (pairwise_dist > lower_bound) & (
            pairwise_dist < upper_bound
        )

        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
            gt_boxes[:, 3] - gt_boxes[:, 1]
        )

        match_matrix = match_matrix.to(torch.float32)
        match_matrix *= 1e8 - gt_areas[:, None]

        match_quality, matched_idxs = match_matrix.max(dim=0)
        matched_idxs[match_quality < 1e-5] = -1

        matched_boxes_this_level = gt_boxes[matched_idxs.clip(min=0)]
        matched_boxes_this_level[matched_idxs < 0, :] = -1

        matched_gt_boxes[level_name] = matched_boxes_this_level

    return matched_gt_boxes

def fcos_get_deltas_from_locations(
    locations: torch.Tensor, gt_boxes: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Compute distances from feature locations to GT box edges. These distances
    are called "deltas" - `(left, top, right, bottom)` or simply `LTRB`. The
    feature locations and GT boxes are given in absolute image co-ordinates.

    These deltas are used as targets for training FCOS to perform box regression
    and centerness regression. They must be "normalized" by the stride of FPN
    feature map (from which feature locations were computed, see the function
    `get_fpn_location_coords`). If GT boxes are "background", then deltas must
    be `(-1, -1, -1, -1)`.

    NOTE: This transformation function should not require GT class label. Your
    implementation must work for GT boxes being `(N, 4)` or `(N, 5)` tensors -
    without or with class labels respectively. You may assume that all the
    background boxes will be `(-1, -1, -1, -1)` or `(-1, -1, -1, -1, -1)`.

    Args:
        locations: Tensor of shape `(N, 2)` giving `(xc, yc)` feature locations.
        gt_boxes: Tensor of shape `(N, 4 or 5)` giving GT boxes.
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            Tensor of shape `(N, 4)` giving deltas from feature locations, that
            are normalized by feature stride.
    """
    deltas = None

    # LTRB = 点到框四边的距离，再除以 stride
    xc = locations[:, 0]
    yc = locations[:, 1]
    left = xc - gt_boxes[:, 0]
    top = yc - gt_boxes[:, 1]
    right = gt_boxes[:, 2] - xc
    bottom = gt_boxes[:, 3] - yc
    deltas = torch.stack([left, top, right, bottom], dim=1) / stride
    # 背景框全是 -1，delta 也写成 -1
    bg = (gt_boxes[:, :4] == -1).all(dim=1)
    deltas[bg] = -1

    return deltas

def fcos_apply_deltas_to_locations(
    deltas: torch.Tensor, locations: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Implement the inverse of `fcos_get_deltas_from_locations` here:

    Given edge deltas (left, top, right, bottom) and feature locations of FPN, get
    the resulting bounding box co-ordinates by applying deltas on locations. This
    method is used for inference in FCOS: deltas are outputs from model, and
    applying them to anchors will give us final box predictions.

    Recall in above method, we were required to normalize the deltas by feature
    stride. Similarly, we have to un-normalize the input deltas with feature
    stride before applying them to locations, because the given input locations are
    already absolute co-ordinates in image dimensions.

    Args:
        deltas: Tensor of shape `(N, 4)` giving edge deltas to apply to locations.
        locations: Locations to apply deltas on. shape: `(N, 2)`
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            Same shape as deltas and locations, giving co-ordinates of the
            resulting boxes `(x1, y1, x2, y2)`, absolute in image dimensions.
    """
    # 推理：delta 先乘回 stride，再从中心推回 xyxy
    d = deltas.clamp(min=0) * stride
    xc = locations[:, 0]
    yc = locations[:, 1]
    x1 = xc - d[:, 0]
    y1 = yc - d[:, 1]
    x2 = xc + d[:, 2]
    y2 = yc + d[:, 3]
    output_boxes = torch.stack([x1, y1, x2, y2], dim=1)

    return output_boxes

def fcos_make_centerness_targets(deltas: torch.Tensor):
    """
    Given LTRB deltas of GT boxes, compute GT targets for supervising the
    centerness regression predictor. See `fcos_get_deltas_from_locations` on
    how deltas are computed. If GT boxes are "background" => deltas are
    `(-1, -1, -1, -1)`, then centerness should be `-1`.

    For reference, centerness equation is available in FCOS paper
    https://arxiv.org/abs/1904.01355 (Equation 3).

    Args:
        deltas: Tensor of shape `(N, 4)` giving LTRB deltas for GT boxes.

    Returns:
        torch.Tensor
            Tensor of shape `(N, )` giving centerness regression targets.
    """
    # 公式：sqrt( min(l,r)/max(l,r) * min(t,b)/max(t,b) )
    l = deltas[:, 0]
    t = deltas[:, 1]
    r = deltas[:, 2]
    b = deltas[:, 3]
    centerness = torch.sqrt(
        (torch.minimum(l, r) / torch.maximum(l, r).clamp(min=1e-8))
        * (torch.minimum(t, b) / torch.maximum(t, b).clamp(min=1e-8))
    )
    bg = (deltas == -1).all(dim=1)
    centerness[bg] = -1

    return centerness

class FCOS(nn.Module):
    """
    FCOS: Fully-Convolutional One-Stage Detector

    This class puts together everything you implemented so far. It contains a
    backbone with FPN, and prediction layers (head). It computes loss during
    training and predicts boxes during inference.
    """

    def __init__(
        self, num_classes: int, fpn_channels: int, stem_channels: List[int]
    ):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = DetectorBackboneWithFPN(out_channels=fpn_channels)
        self.pred_net = FCOSPredictionNetwork(
            num_classes=num_classes, in_channels=fpn_channels,
            stem_channels=stem_channels)

        self._normalizer = 150

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        Args:
            images: Batch of images, tensors of shape `(B, C, H, W)`.
            gt_boxes: Batch of training boxes, tensors of shape `(B, N, 5)`.
                `gt_boxes[i, j] = (x1, y1, x2, y2, C)` gives information about
                the `j`th object in `images[i]`. The position of the top-left
                corner of the box is `(x1, y1)` and the position of bottom-right
                corner of the box is `(x2, x2)`. These coordinates are
                real-valued in `[H, W]`. `C` is an integer giving the category
                label for this bounding box. Not provided during inference.
            test_score_thresh: During inference, discard predictions with a
                confidence score less than this value. Ignored during training.
            test_nms_thresh: IoU threshold for NMS during inference. Ignored
                during training.

        Returns:
            Losses during training and predictions during inference.
        """

        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = None, None, None
        fpn_feats = self.backbone(images)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.pred_net(fpn_feats)

        fpn_feats_shapes = {
            level_name: feat.shape for level_name, feat in fpn_feats.items()
        }
        locations_per_fpn_level = get_fpn_location_coords(
            fpn_feats_shapes, self.backbone.fpn_strides,
            dtype=images.dtype, device=images.device)

        if not self.training:
            return self.inference(
                images, locations_per_fpn_level,
                pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )

        matched_gt_boxes = []
        for i in range(images.shape[0]):
            matched_gt_boxes.append(
                fcos_match_locations_to_gt(
                    locations_per_fpn_level, self.backbone.fpn_strides,
                    gt_boxes[i]))

        matched_gt_deltas = []
        for i in range(images.shape[0]):
            deltas_per_level = {}
            for level_name, locations in locations_per_fpn_level.items():
                deltas_per_level[level_name] = fcos_get_deltas_from_locations(
                    locations, matched_gt_boxes[i][level_name],
                    self.backbone.fpn_strides[level_name])
            matched_gt_deltas.append(deltas_per_level)

        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)

        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)

        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / images.shape[0]
        self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image

        loss_cls, loss_box, loss_ctr = None, None, None

        B = images.shape[0]
        num_locations = matched_gt_boxes.shape[1]

        gt_classes = matched_gt_boxes[:, :, 4].long()
        gt_cls_onehot = torch.zeros_like(pred_cls_logits)
        fg = gt_classes >= 0
        fg_idx = fg.nonzero(as_tuple=True)
        gt_cls_onehot[fg_idx[0], fg_idx[1], gt_classes[fg]] = 1.0
        loss_cls = sigmoid_focal_loss(
            pred_cls_logits, gt_cls_onehot, alpha=0.25, gamma=2.0, reduction="none")

        loss_box = 0.25 * F.l1_loss(
            pred_boxreg_deltas, matched_gt_deltas, reduction="none")
        loss_box[matched_gt_deltas < 0] = 0.0

        gt_centerness = fcos_make_centerness_targets(
            matched_gt_deltas.reshape(-1, 4)).reshape(B, num_locations, 1)
        loss_ctr = F.binary_cross_entropy_with_logits(
            pred_ctr_logits, gt_centerness.clamp(min=0), reduction="none")
        loss_ctr[gt_centerness < 0] = 0.0
        return {
            "loss_cls": loss_cls.sum() / (self._normalizer * images.shape[0]),
            "loss_box": loss_box.sum() / (self._normalizer * images.shape[0]),
            "loss_ctr": loss_ctr.sum() / (self._normalizer * images.shape[0]),
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method. This method
        should not be called from anywhere except from inside `forward`.

        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes.

                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels). Make
                  sure there are no background predictions (-1).

                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions: these values are `sqrt(class_prob * ctrness)`
                  where class_prob and ctrness are obtained by applying sigmoid
                  to corresponding logits.
        """

        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in locations_per_fpn_level.keys():

            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]

            level_pred_boxes, level_pred_classes, level_pred_scores = (
                None,
                None,
                None,
            )

            level_pred_scores = torch.sqrt(
                level_cls_logits.sigmoid_() * level_ctr_logits.sigmoid_()
            )
            level_pred_scores, level_pred_classes = level_pred_scores.max(dim=1)

            keep = level_pred_scores > test_score_thresh
            level_pred_scores = level_pred_scores[keep]
            level_pred_classes = level_pred_classes[keep]
            level_deltas = level_deltas[keep]
            level_locations = level_locations[keep]

            stride = self.backbone.fpn_strides[level_name]
            level_pred_boxes = fcos_apply_deltas_to_locations(
                level_deltas, level_locations, stride)

            H, W = images.shape[2], images.shape[3]
            level_pred_boxes[:, 0].clamp_(min=0, max=W)
            level_pred_boxes[:, 1].clamp_(min=0, max=H)
            level_pred_boxes[:, 2].clamp_(min=0, max=W)
            level_pred_boxes[:, 3].clamp_(min=0, max=H)

            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes)
            pred_scores_all_levels.append(level_pred_scores)

        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        keep = class_spec_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=test_nms_thresh,
        )
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]
        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )
