# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count, stretch):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  return {
    'desired_samples': desired_samples,
    'window_size_samples': window_size_samples,
    'window_stride_samples': window_stride_samples,
    'spectrogram_length': spectrogram_length,
    'dct_coefficient_count': dct_coefficient_count,
    'fingerprint_size': fingerprint_size,
    'label_count': label_count,
    'sample_rate': sample_rate,
    'stretch': stretch
  }


def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None):
  """Builds a model of the requested architecture compatible with the settings.

  There are many possible ways of deriving predictions from a spectrogram
  input, so this function provides an abstract interface for creating different
  kinds of models in a black-box way. You need to pass in a TensorFlow node as
  the 'fingerprint' input, and this should output a batch of 1D features that
  describe the audio. Typically this will be derived from a spectrogram that's
  been run through an MFCC, but in theory it can be any feature vector of the
  size specified in model_settings['fingerprint_size'].

  The function will build the graph it needs in the current TensorFlow graph,
  and return the tensorflow output that will contain the 'logits' input to the
  softmax prediction process. If training flag is on, it will also return a
  placeholder node that can be used to control the dropout amount.

  See the implementations below for the possible model architectures that can be
  requested.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    model_architecture: String specifying which kind of model to create.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
    Exception: If the architecture type isn't recognized.
  """
  if model_architecture == 'single_fc':
    return create_single_fc_model(fingerprint_input, model_settings,
                                  is_training)
  elif model_architecture == 'conv':
    return create_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'resnet':
    return create_resnet(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'multilayer_lstm':
    return create_multilayer_lstm(fingerprint_input, model_settings,
                                  is_training)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "single_fc", "conv",'
                    ' "multilayer_lstm" or "resnet"')


def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)


def create_single_fc_model(fingerprint_input, model_settings, is_training):
  """Builds a model with a single hidden fully-connected layer.

  This is a very simple model with just one matmul and bias layer. As you'd
  expect, it doesn't produce very accurate results, but it is very fast and
  simple, so it's useful for sanity testing.

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  weights = tf.Variable(
    tf.truncated_normal([fingerprint_size, label_count], stddev=0.001))
  bias = tf.Variable(tf.zeros([label_count]))
  logits = tf.matmul(fingerprint_input, weights) + bias
  return logits


def create_conv_model(fingerprint_input, model_settings, is_training):
  """Builds a standard convolutional model.

  This is roughly the network labeled as 'cnn-trad-fpool3' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This produces fairly good quality results, but can involve a large number of
  weight parameters and computations. For a cheaper alternative from the same
  paper with slightly less accuracy, see 'low_latency_conv' below.

  During training, dropout nodes are introduced after each relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_weights = tf.Variable(
    tf.truncated_normal(
      [first_filter_height, first_filter_width, 1, first_filter_count],
      stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  first_dropout = tf.layers.dropout(first_relu, rate=0.5, training=is_training)
  max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_weights = tf.Variable(
    tf.truncated_normal(
      [
        second_filter_height, second_filter_width, first_filter_count,
        second_filter_count
      ],
      stddev=0.01))
  second_bias = tf.Variable(tf.zeros([second_filter_count]))
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                             'SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  second_dropout = tf.layers.dropout(second_relu, rate=0.5, training=is_training)
  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
    second_conv_output_width * second_conv_output_height *
    second_filter_count)
  flattened_second_conv = tf.reshape(second_dropout,
                                     [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
    tf.truncated_normal(
      [second_conv_element_count, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
  return final_fc


def create_resnet(fingerprint_input, model_settings, is_training):
  input_frequency_size = model_settings['dct_coefficient_count']  # 40
  input_time_size = model_settings['spectrogram_length']  # 98
  label_count = model_settings['label_count']  # 12

  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])

  initializer = tf.truncated_normal_initializer(stddev=0.01)

  hidden = fingerprint_4d

  def _ConvBn2d(x, out_channels, kernel_size=3):
    x = tf.layers.conv2d(x, filters=out_channels, kernel_size=kernel_size,
                         padding='SAME', kernel_initializer=initializer)
    x = tf.layers.batch_normalization(x)
    return x

  def _SeScale(x, reduction=16):
    channel = x.shape[-1]
    # x = tf.reduce_mean(x, axis=(1, 2), keep_dims=True)
    x = tf.layers.conv2d(x, filters=reduction, kernel_size=1, padding='SAME',
                         kernel_initializer=initializer)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filters=channel, kernel_size=1, padding='SAME',
                         kernel_initializer=initializer)
    x = tf.nn.sigmoid(x)
    return x

  def _ResBlock(x, out_channels, kernel_size=3):
    z = _ConvBn2d(x, out_channels=out_channels, kernel_size=kernel_size)
    z = tf.nn.relu(z)
    z = _ConvBn2d(z, out_channels=out_channels, kernel_size=kernel_size)
    z = _SeScale(z) * z + x
    z = tf.nn.relu(z)
    return z

  hidden = _ConvBn2d(hidden, out_channels=16, kernel_size=3)
  hidden = tf.nn.relu(hidden)
  hidden = _ResBlock(hidden, out_channels=16, kernel_size=3)
  hidden = tf.layers.max_pooling2d(hidden, pool_size=2, strides=2)

  hidden = tf.layers.dropout(hidden, rate=0.1, training=is_training)
  hidden = _ConvBn2d(hidden, out_channels=32, kernel_size=3)
  hidden = tf.nn.relu(hidden)
  hidden = _ResBlock(hidden, out_channels=32, kernel_size=3)
  hidden = _ResBlock(hidden, out_channels=32, kernel_size=3)
  hidden = tf.layers.max_pooling2d(hidden, pool_size=2, strides=2)

  hidden = tf.layers.dropout(hidden, rate=0.2, training=is_training)
  hidden = _ConvBn2d(hidden, out_channels=64, kernel_size=3)
  hidden = tf.nn.relu(hidden)
  hidden = _ResBlock(hidden, out_channels=64, kernel_size=3)
  hidden = _ResBlock(hidden, out_channels=64, kernel_size=3)
  hidden = tf.layers.max_pooling2d(hidden, pool_size=2, strides=2)

  hidden = tf.layers.dropout(hidden, rate=0.2, training=is_training)
  hidden = _ConvBn2d(hidden, out_channels=128, kernel_size=3)
  hidden = tf.nn.relu(hidden)
  hidden = _ResBlock(hidden, out_channels=128, kernel_size=3)
  hidden = _ResBlock(hidden, out_channels=128, kernel_size=3)

  hidden = tf.layers.dropout(hidden, rate=0.2, training=is_training)
  hidden = _ConvBn2d(hidden, out_channels=256, kernel_size=3)
  hidden = tf.nn.relu(hidden)
  hidden = tf.layers.max_pooling2d(hidden, pool_size=2, strides=2)
  hidden = tf.layers.flatten(hidden)
  hidden = tf.layers.dense(hidden, units=256,
                           kernel_initializer=initializer)
  hidden = tf.nn.relu(hidden)

  hidden = tf.layers.dropout(hidden, rate=0.2, training=is_training)
  final_fc = tf.layers.dense(hidden, units=label_count,
                             kernel_initializer=initializer)
  return final_fc


def create_multilayer_lstm(fingerprint_input, model_settings, is_training):
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  label_count = model_settings['label_count']

  fingerprint_3d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size])

  initializer = tf.truncated_normal_initializer(stddev=0.01)

  cell = tf.nn.rnn_cell.MultiRNNCell([
    tf.nn.rnn_cell.LSTMCell(
      num_units=1024,
      initializer=initializer
    )
    for _ in range(2)
  ])
  _, final_state = tf.nn.dynamic_rnn(
    inputs=fingerprint_3d,
    dtype=tf.float32,
    cell=cell
  )

  hidden0 = tf.concat(final_state[0], axis=1)
  hidden1 = tf.concat(final_state[1], axis=1)
  hidden = tf.concat((hidden0, hidden1), axis=1)

  final_fc = tf.layers.dense(
    inputs=hidden,
    units=label_count,
    kernel_initializer=initializer
  )

  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc
