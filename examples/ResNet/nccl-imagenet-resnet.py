#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet-resnet.py

import argparse
import os

from tensorpack import logger, QueueInput
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.train import (
    TrainConfig, SyncMultiGPUTrainerReplicated, launch_train_with_config)
from tensorpack.dataflow import FakeData
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.gpu import get_nr_gpu

from imagenet_utils import (
    fbresnet_augmentor, get_imagenet_dataflow, ImageNetModel,
    eval_on_ILSVRC12)
from resnet_model import (
    preresnet_group, preresnet_basicblock, preresnet_bottleneck,
    resnet_group, resnet_basicblock, resnet_bottleneck, se_resnet_bottleneck,
    resnet_backbone)
from tensorpack.tfutils.optimizer import AccumGradOptimizerAlt
import tensorflow as tf


class Model(ImageNetModel):
    def __init__(self, depth, data_format='NCHW', mode='resnet', chunk=1):
        super(Model, self).__init__(data_format)

        if mode == 'se':
            assert depth >= 50

        self.mode = mode
        self.chunk = chunk
        basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock
        bottleneck = {
            'resnet': resnet_bottleneck,
            'preact': preresnet_bottleneck,
            'se': se_resnet_bottleneck}[mode]
        self.num_blocks, self.block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]

    def get_logits(self, image):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            return resnet_backbone(
                image, self.num_blocks,
                preresnet_group if self.mode == 'preact' else resnet_group, self.block_func)

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        if self.chunk != 1:
            opt = AccumGradOptimizerAlt(opt, self.chunk)
        return opt


def get_data(name, batch):
    isTrain = name == 'train'
    augmentors = fbresnet_augmentor(isTrain)
    return get_imagenet_dataflow(
        args.data, name, batch, augmentors)


def get_config(model, fake=False):
    nr_tower = max(get_nr_gpu(), 1)
    assert args.logical_batch % args.chunk == 0
    physical_batch = args.logical_batch / args.chunk
    assert physical_batch % nr_tower == 0
    batch = physical_batch // nr_tower

    if fake:
        logger.info("For benchmark, batch size is fixed to 64 per tower.")
        dataset_train = FakeData(
            [[64, 224, 224, 3], [64]], 1000, random=False, dtype='uint8')
        callbacks = []
    else:
        logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
        dataset_train = get_data('train', batch)
        dataset_val = get_data('val', batch)

        BASE_LR = 0.1 * (args.logical_batch / 256.0)
        callbacks = [
            ModelSaver(),
            EstimatedTimeLeft(),
            ScheduledHyperParamSetter(
                'learning_rate', [(10, BASE_LR * 1e-1), (20, BASE_LR * 1e-2),
                                  (90, BASE_LR * 1e-3), (100, BASE_LR * 1e-4)]),
        ]
        if BASE_LR > 0.1:
            warmup_steps = 5*(1281167//physical_batch)
            logger.info("learning_rate growth from 0.1 to {} during first {} steps".format(BASE_LR, warmup_steps))
            callbacks.append(
                ScheduledHyperParamSetter(
                    'learning_rate', [(0, 0.1), (warmup_steps, BASE_LR)], interp='linear', step_based=True))

        infs = [ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')]
        if nr_tower == 1:
            # single-GPU inference with queue prefetch
            callbacks.append(InferenceRunner(QueueInput(dataset_val), infs, one_liner=True))
        else:
            # multi-GPU inference (with mandatory queue prefetch)
            callbacks.append(DataParallelInferenceRunner(
                dataset_val, infs, list(range(nr_tower)), one_liner=True))


    return TrainConfig(
        model=model,
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=100 if args.fake else 1281167 // physical_batch,
        max_epoch=args.epoch,
        one_liner=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--fake', help='use fakedata to test or benchmark this model', action='store_true')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NCHW')
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=50, choices=[18, 34, 50, 101, 152])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--logical_batch', default=256, type=int,
                        help='total batch size. 32 per GPU gives best accuracy, higher values should be similarly good')
    parser.add_argument('--mode', choices=['resnet', 'preact', 'se'],
                        help='variants of resnet to use', default='resnet')
    parser.add_argument('--chunk', help='accumulation', type=int, default=1)
    parser.add_argument('--epoch', help='total epoch', type=int, default=30)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = Model(args.depth, args.data_format, args.mode, chunk=args.chunk)
    if args.eval:
        batch = 128    # something that can run on one gpu
        ds = get_data('val', batch)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    else:
        if args.fake:
            logger.set_logger_dir(os.path.join('train_log', 'tmp'), 'd')
        else:
            logger.set_logger_dir(
                os.path.join('train_log', 'imagenet-{}-d{}'.format(args.mode, args.depth)), action='d')

        config = get_config(model, fake=args.fake)
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SyncMultiGPUTrainerReplicated(max(get_nr_gpu(), 1), mode="nccl")
        launch_train_with_config(config, trainer)
