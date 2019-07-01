import logging

import torch.nn as nn

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

    def simple_test(self, img, img_meta):
        x = self.backbone(img)
        img_size = img_meta['img_shape']
        seg_pred = self.decoder(x, img_size)
        return seg_pred.argmax(1)

    def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)
