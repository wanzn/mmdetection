import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import constant_init, xavier_init
from ..utils import ActivatedBatchNorm
from ..losses import lovasz_losses as L
# from ..builder import build_loss
# from ..losses import accuracy


class DecoderBase(nn.Module):
    def __init__(self,
                 module,
                 in_channels,
                 num_classes=4,
                 num_filters=16,
                 loss_seg=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0
                 )):
        super(DecoderBase, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.center = module(in_channels[4],
                             num_filters * 32 * 2,
                             num_filters * 32)
        self.decoder5 = module(in_channels[4] + num_filters * 32,
                               num_filters * 32 * 2,
                               num_filters * 16)
        self.decoder4 = module(in_channels[3] + num_filters * 16,
                               num_filters * 16 * 2,
                               num_filters * 8)
        self.decoder3 = module(in_channels[2] + num_filters * 8,
                               num_filters * 8 * 2,
                               num_filters * 4)
        self.decoder2 = module(in_channels[1] + num_filters * 4,
                               num_filters * 4 * 2,
                               num_filters * 2)
        self.decoder1 = module(in_channels[0] + num_filters * 2,
                               num_filters * 2 * 2,
                               num_filters)

        self.logits = nn.Sequential(
            nn.Conv2d(num_filters * 31, 64, kernel_size=1, padding=0),
            ActivatedBatchNorm(64),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        self.loss_seg = L.lovasz_softmax

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self, x, img_size):
        assert len(x) == 5
        e1, e2, e3, e4, e5 = x

        c = self.center(self.pool(e5))
        e1_up = F.interpolate(
            e1, scale_factor=2, mode='bilinear', align_corners=False)

        d5 = self.decoder5(c, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1_up)

        u5 = F.interpolate(d5, img_size, mode='bilinear', align_corners=False)
        u4 = F.interpolate(d4, img_size, mode='bilinear', align_corners=False)
        u3 = F.interpolate(d3, img_size, mode='bilinear', align_corners=False)
        u2 = F.interpolate(d2, img_size, mode='bilinear', align_corners=False)

        # Hyper column
        d = torch.cat((d1, u2, u3, u4, u5), 1)
        logits = self.logits(d)

        return logits

    def loss(self, seg_score, gt_labels):
        losses = dict()
        seg_score = seg_score.softmax(dim=1)
        gt_labels = gt_labels.squeeze(1).long()
        losses['loss_seg'] = self.loss_seg(seg_score, gt_labels)
        return losses
