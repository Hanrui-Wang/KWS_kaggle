from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

import input_data
import models

FLAGS = None


def main(_):
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)
  
  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()
  
  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
  model_settings = models.prepare_model_settings(
    len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
    FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
    FLAGS.window_stride_ms, FLAGS.dct_coefficient_count, 0)
  audio_processor = input_data.AudioProcessor(
    FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
    FLAGS.unknown_percentage,
    FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
    FLAGS.testing_percentage, model_settings)
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
  # Figure out the learning rates for each training phase. Since it's often
  # effective to have high learning rates at the start of training, followed by
  # lower levels towards the end, the number of steps and learning rates can be
  # specified as comma-separated lists to define the rate at each stage. For
  # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
  # will run 13,000 training loops in total, with a rate of 0.001 for the first
  # 10,000, and 0.0001 for the final 3,000.
  training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
      '--how_many_training_steps and --learning_rate must be equal length '
      'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                 len(learning_rates_list)))
  
  fingerprint_input = tf.placeholder(
    tf.float32, [None, fingerprint_size], name='fingerprint_input')
  is_training = tf.placeholder(tf.bool)
  
  logits = models.create_model(
    fingerprint_input,
    model_settings,
    FLAGS.model_architecture,
    is_training)
  
  # Define loss and optimizer
  ground_truth_input = tf.placeholder(
    tf.int64, [None], name='groundtruth_input')
  
  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]
  
  # Create the back propagation and training evaluation machinery in the graph.
  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
      labels=ground_truth_input, logits=logits)
  
  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    learning_rate_input = tf.placeholder(
      tf.float32, [], name='learning_rate_input')
    train_step = tf.train.AdamOptimizer(
      learning_rate_input).minimize(cross_entropy_mean)
  predicted_indices = tf.argmax(logits, 1)
  correct_prediction = tf.equal(predicted_indices, ground_truth_input)
  confusion_matrix = tf.confusion_matrix(
    ground_truth_input, predicted_indices, num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  global_step = tf.train.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)
  
  saver = tf.train.Saver(tf.global_variables())
  
  tf.global_variables_initializer().run()
  
  start_step = 1
  
  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)
  
  audio_processor.prepare_LB_test(FLAGS.LB_test_set_path)
  
  prd = (
    'silence', 'unknown', 'yes', 'no', 'up', 'down', 'left', 'right', 'on',
    'off',
    'stop', 'go')
  set_size = audio_processor.set_size('LB_test')
  # set_size = 10000
  tf.logging.info('LB_test_set_size=%d', set_size)
  fid_w = open('prediction-' + FLAGS.model_architecture + '.txt', 'w')
  fid_s = open('scores-' + FLAGS.model_architecture + '.txt', 'w')
  ii = 0
  for i in xrange(0, set_size, FLAGS.LB_test_batch_size):
    print(ii * FLAGS.LB_test_batch_size)
    ii += 1
    test_fingerprints, test_ground_truth = audio_processor.get_data(
      FLAGS.LB_test_batch_size, i, model_settings, 0.0, 0.0, 0, 'LB_test', sess)
    test_accuracy, scores, prediction, = sess.run(
      [evaluation_step, logits, predicted_indices],
      feed_dict={
        fingerprint_input: test_fingerprints,
        ground_truth_input: test_ground_truth,
        is_training: False
      })
    batch_size = min(FLAGS.LB_test_batch_size, set_size - i)
    for j in range(0, batch_size):
      fid_w.write("%s\n" % prd[prediction[j]])
      fid_s.write("%s\n" % ' '.join(scores[j]))
      # print(prediction[j])
  fid_w.close()
  
  # write submission csv file
  fid_filename = open('LB_test_filename.txt', 'r')
  fid_prediction = open('prediction' + FLAGS.model_architecture + '.txt', 'r')
  
  filename = fid_filename.read().split('\n')
  prediction = fid_prediction.read().split('\n')
  
  fid_filename.close()
  fid_prediction.close()
  
  fid_w = open(FLAGS.submission_file_name, 'w')
  
  fid_w.write("fname,label\n")
  k = len(filename)
  for i in range(0, k):
    fid_w.write("%s,%s\n" % (filename[i], prediction[i]))
  fid_w.close()
  fid_s.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--data_url',
    type=str,
    # pylint: disable=line-too-long
    default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
    # pylint: enable=line-too-long
    help='Location of speech training data archive on the web.')
  parser.add_argument(
    '--data_dir',
    type=str,
    default='/tmp/speech_dataset/',
    help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
    '--background_volume',
    type=float,
    default=0.1,
    help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
    '--background_frequency',
    type=float,
    default=0.8,
    help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
    '--silence_percentage',
    type=float,
    default=10.0,
    help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
    '--unknown_percentage',
    type=float,
    default=10.0,
    help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
    '--time_shift_ms',
    type=float,
    default=100.0,
    help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
    '--testing_percentage',
    type=int,
    default=10,
    help='What percentage of wavs to use as a test set.')
  parser.add_argument(
    '--validation_percentage',
    type=int,
    default=10,
    help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
    '--sample_rate',
    type=int,
    default=16000,
    help='Expected sample rate of the wavs', )
  parser.add_argument(
    '--clip_duration_ms',
    type=int,
    default=1000,
    help='Expected duration in milliseconds of the wavs', )
  parser.add_argument(
    '--window_size_ms',
    type=float,
    default=30.0,
    help='How long each spectrogram timeslice is', )
  parser.add_argument(
    '--window_stride_ms',
    type=float,
    default=10.0,
    help='How long each spectrogram timeslice is', )
  parser.add_argument(
    '--dct_coefficient_count',
    type=int,
    default=40,
    help='How many bins to use for the MFCC fingerprint', )
  parser.add_argument(
    '--how_many_training_steps',
    type=str,
    default='15000,3000',
    help='How many training loops to run', )
  parser.add_argument(
    '--eval_step_interval',
    type=int,
    default=400,
    help='How often to evaluate the training results.')
  parser.add_argument(
    '--learning_rate',
    type=str,
    default='0.001,0.0001',
    help='How large a learning rate to use when training.')
  parser.add_argument(
    '--batch_size',
    type=int,
    default=100,
    help='How many items to train with at once', )
  parser.add_argument(
    '--summaries_dir',
    type=str,
    default='/tmp/retrain_logs',
    help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
    '--wanted_words',
    type=str,
    default='yes,no,up,down,left,right,on,off,stop,go',
    help='Words to use (others will be added to an unknown label)', )
  parser.add_argument(
    '--train_dir',
    type=str,
    default='/tmp/speech_commands_train',
    help='Directory to write event logs and checkpoint.')
  parser.add_argument(
    '--save_step_interval',
    type=int,
    default=100,
    help='Save model checkpoint every save_steps.')
  parser.add_argument(
    '--start_checkpoint',
    type=str,
    default='',
    help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
    '--model_architecture',
    type=str,
    default='conv',
    help='What model architecture to use')
  parser.add_argument(
    '--check_nans',
    type=bool,
    default=False,
    help='Whether to check for invalid numbers during processing')
  parser.add_argument(
    '--LB_test_set_path',
    type=str,
    default='',
    help='The path to Leaderboard test set')
  parser.add_argument(
    '--LB_test_batch_size',
    type=int,
    default=200,
    help='Leaderboard test set batch size')
  parser.add_argument(
    '--submission_file_name',
    type=str,
    default='submission00.csv',
    help='Leaderboard submission file name')
  
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
