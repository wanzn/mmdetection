import torch.utils.checkpoint as cp

import torch.nn as nn
from .senet import SEModule
from .resnext import Bottleneck as _Bottleneck, ResNeXt
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer


class SEResNextBottleneck(_Bottleneck):

    def __init__(self, reduction=16, **kwargs):
        super(SEResNextBottleneck, self).__init__(**kwargs)
        self.se_module = SEModule(self.planes * 4, reduction=reduction)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if not self.with_dcn:
                out = self.conv2(out)
            elif self.with_modulated_dcn:
                offset_mask = self.conv2_offset(out)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                out = self.conv2(out, offset, mask)
            else:
                offset = self.conv2_offset(out)
                out = self.conv2(out, offset)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_gen_attention:
                out = self.gen_attention_block(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_gcb:
                out = self.context_block(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.se_module(out)
            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(reduction,
                   block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   groups=1,
                   base_width=4,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   dcn=None,
                   gcb=None):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1],
        )

    layers = []
    layers.append(
        block(
            reduction=reduction,
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            groups=groups,
            base_width=base_width,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn,
            gcb=gcb))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                reduction=reduction,
                inplanes=inplanes,
                planes=planes,
                stride=1,
                dilation=dilation,
                groups=groups,
                base_width=base_width,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class SEResNext(ResNeXt):

    arch_settings = {
        50: (SEResNextBottleneck, (3, 4, 6, 3)),
        101: (SEResNextBottleneck, (3, 4, 23, 3))
    }

    def __init__(self, reduction=16, **kwargs):
        super(SEResNext, self).__init__(**kwargs)
        self.inplanes = 64
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            gcb = self.gcb if self.stage_with_gcb[i] else None
            planes = 64 * 2**i
            res_layer = make_res_layer(
                reduction,
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                groups=self.groups,
                base_width=self.base_width,
                style=self.style,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dcn=dcn,
                gcb=gcb)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()
