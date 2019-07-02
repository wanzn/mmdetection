import os
import os.path as osp

import mmcv
import numpy as np
import cv2 as cv
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from mmdet.datasets.transforms import ImageTransform, SegMapTransform, Numpy2Tensor
from mmdet.datasets.utils import to_tensor, random_scale
from mmdet.datasets.extra_aug import ExtraAugmentation


class JingweiDataset(Dataset):

    CLASSES = ('cured tobacco', ' corn', 'barley rice')

    def __init__(self,
                 img_prefix,
                 split_file,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,
                 flip_ratio=0,
                 with_semantic_seg=True,
                 ann_file=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 proposal_file=None):
        self.img_prefix = img_prefix
        self.num_classes = len(self.CLASSES) + 1

        if not test_mode:
            # self.label_path = img_prefix + '/crop_label'

            with open(osp.join(img_prefix, '..', split_file)) as f:
                self.img_infos = f.read().splitlines()
        else:
            self.img_infos = os.listdir(img_prefix)

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for match)
        self.size_divisor = size_divisor

        # with semantic segmentation (stuff) annotation or not
        self.with_seg = with_semantic_seg
        # prefix of semantic segmentation map path
        self.ann_file = ann_file
        # rescale factor for segmentation maps
        self.seg_scale_factor = seg_scale_factor
        # in test mode or not
        self.test_mode = test_mode

        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.seg_transform = SegMapTransform(self.size_divisor)
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio
        self._set_group_flag()

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if int(img_info[-5]) // 2 == 1:
                self.flag[i] = 1

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def get_ann_mask(self, idx):
        img_info = self.img_infos[idx]
        label_path = os.path.join(self.seg_prefix, img_info)
        label_img = cv.imread(label_path, flag='unchanged')

        return label_img

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info))
        ori_shape = (img.shape[0], img.shape[1], 3)
        # extra augmentation
        # if self.extra_aug is not None:
        #     img = self.extra_aug(img)
        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.ann_file, img_info),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        # h, w = gt_seg.shape
        # _gt_seg = gt_seg.ravel()
        # gt_seg_one_hot = np.zeros((h*w, self.num_classes), dtype=np.uint8)
        # gt_seg_one_hot[np.arange(h*w),  _gt_seg] = 1
        # gt_seg_one_hot = gt_seg_one_hot.reshape(h, w, self.num_classes)
        # gt_seg_one_hot = gt_seg_one_hot.transpose(2, 0, 1)

        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True))
        if self.with_seg:
            data['gt_labels'] = DC(to_tensor(gt_seg), stack=True)
        return data

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info))

        def prepare_single(img, scale, flip):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            return _img, _img_meta

        imgs = []
        img_metas = []
        for scale in self.img_scales:
            _img, _img_meta = prepare_single(img, scale, False)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            if self.flip_ratio > 0:
                _img, _img_meta = prepare_single(img, scale, True)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
        data = dict(img=imgs, img_meta=img_metas)
        return data


if __name__ == "__main__":
    data = JingweiDataset(img_prefix='data/jingwei/jingwei_round1_train_20190619/crop',
                          split_file='split_train.txt',
                          img_scale=(512, 512),
                          img_norm_cfg=dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                        multiscale_mode='value',
                        size_divisor=None,
                        flip_ratio=0,
                        with_semantic_seg=True,
                        ann_file='data/jingwei/jingwei_round1_train_20190619/crop_label',
                        seg_scale_factor=1,
                        extra_aug=None,
                        resize_keep_ratio=True,
                        test_mode=False)
    a = data[0]
    print(a)
