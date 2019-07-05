import logging

import numpy as np
import torch.nn as nn
import cv2 as cv

from mmdet.core import tensor2imgs
from .. import builder
from ..registry import DETECTORS


@DETECTORS.register_module
class UNet(nn.Module):
    def __init__(self,
                 backbone,
                 decoder,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(UNet, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.decoder = builder.build_decoder(decoder)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))
        self.backbone.init_weights(pretrained=pretrained)
        self.decoder.init_weights()

    def forward_train(self, img, img_meta, gt_labels):
        x = self.backbone(img)
        img_size = img_meta[0]['img_shape'][:2]
        seg_pred = self.decoder(x, img_size)
        losses = dict()
        loss_seg = self.decoder.loss(seg_pred, gt_labels)
        losses.update(loss_seg)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.backbone(img)
        img_size = img_meta[0]['img_shape'][:2]
        seg_pred = self.decoder(x, img_size)
        return seg_pred.argmax(1)

    def aug_test(self, img, img_meta, rescale=False):
        raise NotImplementedError

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def show_result(self, data, result, img_norm_cfg):
        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)

        result = result.squeeze(0).cpu().numpy()
        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            for j in range(1, 3):
                color_mask = np.random.randint(
                    0, 256, (1, 3), dtype=np.uint8)
                mask = result == j
                img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
                cv.imshow('out', img_show)
                cv.waitKey(0)
                cv.destroyAllWindows()
