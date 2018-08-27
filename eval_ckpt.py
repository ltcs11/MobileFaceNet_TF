# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Tensorflow implementation for MobileFaceNet.
Author: aiboy.wei@outlook.com .
'''

from utils.data_process import parse_function, load_data
from tensorflow.core.protobuf import config_pb2
from nets.MobileFaceNet import inference
from losses.face_losses import cos_loss
from verification import evaluate
from scipy.optimize import brentq
from utils.common import train
from scipy import interpolate
from datetime import datetime
from sklearn import metrics
import tensorflow as tf
import numpy as np
import argparse
import time
import os

slim = tf.contrib.slim



def get_parser():
  parser = argparse.ArgumentParser(description='parameters to evaluate the ckpt')
  parser.add_argument('--image_size', default=[112, 112], help='the image size')
  parser.add_argument('--embedding_size', type=int, help='Dimensionality of the embedding.', default=128)
  parser.add_argument('--test_batch_size', type=int, help='Number of images to process in a batch in the test set.', default=64)
  # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
  parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
  parser.add_argument('--eval_db_path', default='./datasets', help='evluate datasets base path')
  parser.add_argument('--eval_nrof_folds', type=int, help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
  parser.add_argument('--ckpt_best_path', default='./ckpt_best', help='the ckpt file you want to evaluate')
  parser.add_argument('--eval_file_path', default='./ckpt_best', help='the ckpt evaluate result save path')
  parser.add_argument('--pretrained_model', type=str, default='./ckpt_best', help='Load a pretrained model before training starts.')
  parser.add_argument('--ckpt_index_list', default=['MobileFaceNet_iter_165000.ckpt'],help='ckpt file index.')
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  with tf.Graph().as_default():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = get_parser()
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    result_dir = os.path.join(os.path.expanduser(args.eval_file_path), subdir)
    if not os.path.isdir(result_dir):  # Create the log directory if it doesn't exist
      os.makedirs(result_dir)

  # prepare validate datasets
  ver_list = []
  ver_name_list = []
  for db in args.eval_datasets:
    print('begin db %s convert.' % db)
    data_set = load_data(db, args.image_size, args)
    ver_list.append(data_set)
    ver_name_list.append(db)

  # pretrained model path
  pretrained_model = os.path.expanduser(args.pretrained_model)

  inputs = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)
  labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
  phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=None, name='phase_train')

  w_init_method = slim.initializers.xavier_initializer()
  prelogits, net_points = inference(inputs, phase_train=phase_train_placeholder)
  embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

  sess = tf.Session()
  saver = tf.train.Saver()

  for file_index in args.ckpt_index_list:
    path = pretrained_model + '/' + file_index
    print('Pre-trained model: %s' % path)
    saver.restore(sess, path)
    for ver_step in range(len(ver_list)):
      start_time = time.time()
      data_sets, issame_list = ver_list[ver_step]
      emb_array = np.zeros((data_sets.shape[0], args.embedding_size))
      nrof_batches = data_sets.shape[0] // args.test_batch_size
      for index in range(nrof_batches):  # actual is same multiply 2, test data total
        start_index = index * args.test_batch_size
        end_index = min((index + 1) * args.test_batch_size, data_sets.shape[0])

        feed_dict = {inputs: data_sets[start_index:end_index, ...], phase_train_placeholder: False}
        emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

      tpr, fpr, accuracy, val, val_std, far = evaluate(emb_array, issame_list, nrof_folds=args.eval_nrof_folds)
      duration = time.time() - start_time

      print("total time %.3f to evaluate %d images of %s" % (duration, data_sets.shape[0], ver_name_list[ver_step]))
      print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
      print('fpr and tpr: %1.3f %1.3f' % (np.mean(fpr, 0), np.mean(tpr, 0)))
      print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

      auc = metrics.auc(fpr, tpr)
      print('Area Under Curve (AUC): %1.3f' % auc)
      eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
      print('Equal Error Rate (EER): %1.3f\n' % eer)

      with open(os.path.join(result_dir, '{}_result.txt'.format(ver_name_list[ver_step])), 'at') as f:
        f.write('%.5f\t%.5f\t%1.3f\t%1.3f\n' % (np.mean(accuracy), val, np.mean(fpr, 0), np.mean(tpr, 0)))

  sess.close()


