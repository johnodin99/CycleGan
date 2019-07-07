""" Freeze variables and convert 2 generator networks to 2 GraphDef files.
This makes file size smaller and can be used for inference in production.
An example of command-line usage is:
python export_graph.py --checkpoint_dir checkpoints/20170424-1152 \
                       --XtoY_model apple2orange.pb \
                       --YtoX_model orange2apple.pb \
                       --image_size 256
"""

import tensorflow as tf
import os
from tensorflow.python.tools.freeze_graph import freeze_graph
from model import CycleGAN
import utils
import shutil

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir_by_time', r'./checkpoints/20190707-2232', 'checkpoints directory path')
tf.flags.DEFINE_string('checkpoint_dir', r'./checkpoints', 'checkpoints directory path')
tf.flags.DEFINE_string('XtoY_model', 'Good2Bad.pb', 'XtoY model name, default: Good2Bad.pb')
tf.flags.DEFINE_string('YtoX_model', 'Bad2Good.pb', 'YtoX model name, default: Bad2Good.pb')
tf.flags.DEFINE_integer('image_size', '400', 'image size, default: 400')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')

files = os.listdir(FLAGS.checkpoint_dir_by_time)

for f in files:
  shutil.copy(os.path.join(FLAGS.checkpoint_dir_by_time, f), os.path.join(FLAGS.checkpoint_dir, f))

def export_graph(model_name, XtoY=True):
  graph = tf.Graph()

  with graph.as_default():
    cycle_gan = CycleGAN(ngf=FLAGS.ngf, norm=FLAGS.norm, image_size=FLAGS.image_size)

    input_image = tf.placeholder(tf.float32, shape=[FLAGS.image_size, FLAGS.image_size, 3], name='input_image')
    cycle_gan.model()
    if XtoY:
      output_image = cycle_gan.G.sample(tf.expand_dims(input_image, 0))
    else:
      output_image = cycle_gan.F.sample(tf.expand_dims(input_image, 0))

    output_image = tf.identity(output_image, name='output_image')

    restore_saver = tf.train.Saver()

    export_saver = tf.train.Saver()


  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    ckpt_list = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir).all_model_checkpoint_paths

    for m in ckpt_list:

      print("ckpt: "+m)

      temp=m.split("-")

      iteration=temp[1]

      restore_saver.restore(sess, m)

      output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [output_image.op.name])

      tf.train.write_graph(output_graph_def, 'pretrained', str(iteration)+"_"+model_name, as_text=False)



def main(unused_argv):

  #print('Export XtoY model...')
  export_graph(FLAGS.XtoY_model, XtoY=True)
  #print('Export YtoX model...')
  #export_graph(FLAGS.YtoX_model, XtoY=False)

if __name__ == '__main__':
  tf.app.run()
