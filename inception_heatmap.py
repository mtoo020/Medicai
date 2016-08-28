from scipy import misc
import numpy as np
import tensorflow as tf

#from inception import image_processing
from inception import inception_model as inception

#from PIL import Image
#from PIL import ImageFile


import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_file', '',
                           """Image to segment.""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")


# def _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op):
#   """Runs Eval once.
#
#   Args:
#     saver: Saver.
#     summary_writer: Summary writer.
#     top_1_op: Top 1 op.
#     top_5_op: Top 5 op.
#     summary_op: Summary op.
#   """
#   with tf.Session() as sess:
#     ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
#     if ckpt and ckpt.model_checkpoint_path:
#       if os.path.isabs(ckpt.model_checkpoint_path):
#         # Restores from checkpoint with absolute path.
#         saver.restore(sess, ckpt.model_checkpoint_path)
#       else:
#         # Restores from checkpoint with relative path.
#         saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
#                                          ckpt.model_checkpoint_path))
#
#       # Assuming model_checkpoint_path looks something like:
#       #   /my-favorite-path/imagenet_train/model.ckpt-0,
#       # extract global_step from it.
#       global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#       print('Succesfully loaded model from %s at step=%s.' %
#             (ckpt.model_checkpoint_path, global_step))
#     else:
#       print('No checkpoint file found')
#       return
#
#     # Start the queue runners.
#     coord = tf.train.Coordinator()
#     try:
#       threads = []
#       for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
#         threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
#                                          start=True))
#
#       num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
#       # Counts the number of correct predictions.
#       count_top_1 = 0.0
#       count_top_5 = 0.0
#       total_sample_count = num_iter * FLAGS.batch_size
#       step = 0
#
#       print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
#       start_time = time.time()
#       while step < num_iter and not coord.should_stop():
#         top_1, top_5 = sess.run([top_1_op, top_5_op])
#         count_top_1 += np.sum(top_1)
#         count_top_5 += np.sum(top_5)
#         step += 1
#         if step % 20 == 0:
#           duration = time.time() - start_time
#           sec_per_batch = duration / 20.0
#           examples_per_sec = FLAGS.batch_size / sec_per_batch
#           print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
#                 'sec/batch)' % (datetime.now(), step, num_iter,
#                                 examples_per_sec, sec_per_batch))
#           start_time = time.time()
#
#       # Compute precision @ 1.
#       precision_at_1 = count_top_1 / total_sample_count
#       recall_at_5 = count_top_5 / total_sample_count
#       print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
#             (datetime.now(), precision_at_1, recall_at_5, total_sample_count))
#
#       summary = tf.Summary()
#       summary.ParseFromString(sess.run(summary_op))
#       summary.value.add(tag='Precision @ 1', simple_value=precision_at_1)
#       summary.value.add(tag='Recall @ 5', simple_value=recall_at_5)
#       summary_writer.add_summary(summary, global_step)
#
#     except Exception as e:  # pylint: disable=broad-except
#       coord.request_stop(e)
#
#     coord.request_stop()
#     coord.join(threads, stop_grace_period_secs=10)

def evaluate():
    #ImageFile.LOAD_TRUNCATED_IMAGES = True

    #im = Image.open(FLAGS.image_file)
    im = misc.imread(FLAGS.image_file)
    print FLAGS.image_file
    w = im.shape[0]
    h = im.shape[1]
    a = im # np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
    print a.shape
    a = np.pad(a,((150,150),(150,150),(0,0)),mode='constant',constant_values=[0])

    num_classes = 1001

    """Runs Eval once.

    Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_1_op: Top 1 op.
    top_5_op: Top 5 op.
    summary_op: Summary op.
    """
    with tf.Graph().as_default():
        images = tf.placeholder(dtype=tf.float32,shape=(1, 299, 299, 3))
        logits, _ = inception.inference(images, num_classes)

        # Restore the moving average version of the learned variables for eval.

        with tf.Session() as sess:
            variable_averages = tf.train.ExponentialMovingAverage(
                inception.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver() #variables_to_restore)
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                if os.path.isabs(ckpt.model_checkpoint_path):
                    # Restores from checkpoint with absolute path.
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    # Restores from checkpoint with relative path.
                    saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,ckpt.model_checkpoint_path))
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Succesfully loaded model from %s at step=%s.' % (ckpt.model_checkpoint_path, global_step))
            else:
                print('No checkpoint file found')
                return

            for i in range(0,w,64):
                for j in range(0,h,64):
                    crp = a[i:i+299, j:j+299, :].reshape(1, 299, 299, 3)
                    crp = (crp.astype('float32')/128.0) - 1
                    #logits, _ = inception.inference(imgs, num_classes)
                    l = sess.run(logits, feed_dict={images:crp})
                    #print crp[0:4,0:4,:]
                    print np.argmax(l)


    #
    # # Restore the moving average version of the learned variables for eval.
    # variable_averages = tf.train.ExponentialMovingAverage(
    #     inception.MOVING_AVERAGE_DECAY)
    # variables_to_restore = variable_averages.variables_to_restore()
    # saver = tf.train.Saver(variables_to_restore)
    #
    # # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.merge_all_summaries()
    #
    # graph_def = tf.get_default_graph().as_graph_def()
    # summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
    #                                         graph_def=graph_def)
    #
    # while True:
    #   _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op)
    #   if FLAGS.run_once:
    #     break
    #   time.sleep(FLAGS.eval_interval_secs)

def main(unused_argv=None):
    evaluate()

if __name__ == '__main__':
  tf.app.run()
