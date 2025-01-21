# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class Detection3DPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolo11n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds, preds_3d = self.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            max_det=self.args.max_det
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, pred_3d, orig_img, img_path in zip(preds, preds_3d, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred, result_3d=pred_3d))
        return results

    @staticmethod
    def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=300,
        max_nms=30000,
        max_wh=7680,
        in_place=False,
    ):
        """
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Args:
            prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            conf_thres (float): The confidence threshold below which boxes will be filtered out.
                Valid values are between 0.0 and 1.0.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
                Valid values are between 0.0 and 1.0.
        Returns:
            (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
                shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
                (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        """
        import torchvision  # scope for faster 'import ultralytics'
        import torch
        from ultralytics.utils.ops import xywh2xyxy

        # Checks
        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
        pred_2d, pred_3d = prediction
        if isinstance(pred_2d, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = pred_2d[0]  # select only inference output

        bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
        nc = prediction.shape[1] - 4  # number of classes
        xc = prediction[:, 4:].amax(1) > conf_thres  # candidates

        prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
        pred_3d = pred_3d.transpose(-1, -2)
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

        output = [torch.zeros((0, 6), device=prediction.device)] * bs
        output_3d = [torch.zeros((0, 8), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            x_3d = pred_3d[xi][xc[xi]]

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls = x.split((4, nc), 1)

            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * max_wh  # classes
            scores = x[:, 4]  # scores
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections

            # # Experimental
            # merge = False  # use merge-NMS
            # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            #     # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            #     from .metrics import box_iou
            #     iou = box_iou(boxes[i], boxes) > iou_thres  # IoU matrix
            #     weights = iou * scores[None]  # box weights
            #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            #     redundant = True  # require redundant detections
            #     if redundant:
            #         i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            output_3d[xi] = torch.cat((x[i], x_3d[i]), dim=-1)

        return output, output_3d