"""Evaluates a model against clean examples both train and test as specified
   in config.json"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

import cifar10_input
from model import Model


def run_eval(checkpoint, train=False, num_eval_examples=None, data_dir=None):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    model = Model(mode='eval')
    saver = tf.train.Saver()
    eval_batch_size = 64
    total_corr = 0

    cifar = cifar10_input.CIFAR10Data(data_dir)

    if train:
        x_nat = cifar.train_data.xs
        y_nat = cifar.train_data.ys
    else:
        x_nat = cifar.eval_data.xs
        y_nat = cifar.eval_data.ys

    if num_eval_examples is None:
        num_eval_examples = len(x_nat)

    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    y_pred = []  # label accumulator
    red_neck = []

    with tf.Session() as sess:
        # Restore the checkpoint
        saver.restore(sess, checkpoint)

        # Iterate over the samples batch-by-batch
        for ibatch in tqdm(range(num_batches)):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)

            x_batch = x_nat[bstart:bend, :]
            y_batch = y_nat[bstart:bend]
            dict_clean = {model.x_input: x_batch,
                          model.y_input: y_batch}
            cur_corr, y_pred_batch, neck = sess.run([model.num_correct, model.y_pred, model.neck],
                                                    feed_dict=dict_clean)

            total_corr += cur_corr
            y_pred.append(y_pred_batch)
            red_neck.append(neck)
            # print('ibatch: ', ibatch, '/', num_batches)

    accuracy = total_corr / num_eval_examples

    print('Accuracy: {:.2f}%'.format(100.0 * accuracy))
    test_str = 'test'
    if train:
        test_str = 'train'
    np.save(sys.argv[1] + test_str + '_red_neck.npy', np.concatenate(red_neck,
                                                                     axis=0))
    # print('Output saved at pred.npy')


if __name__ == '__main__':
    import json

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_dir = config['model_dir']
    if len(sys.argv) > 1:
        model_dir = 'models/' + sys.argv[1]

    data_path = config['data_path']
    do_train = False
    if len(sys.argv) > 2:
        do_train = sys.argv[2].lower() == 'true'

    checkpoint = tf.train.latest_checkpoint(model_dir)

    if checkpoint is None:
        print('No checkpoint found')

    print('running: ', do_train, model_dir)
    run_eval(checkpoint, train=do_train, data_dir=data_path)
