import tensorflow as tf
import numpy as np
import custom_ops as ops

from wrn import residual_block, _res_add


class ResNet(object):
  def __init__(self, sess, dataset):
	self.sess = sess
    self.dataset = dataset
    self.batch_size = 128
    self.input_height = 32
    self.input_width = 32
    self.num_input = 3 * 32 * 32
    self.num_steps = 10000
    self.num_classes = 10
    self.temperature = 5
    self.display_step = 100
    self.learning_rate = 0.0001
    self.checkpoint_dir = "/tmp/training/wrn/"
    self.checkpoint_file = "/student"
    self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file + ".cpkt")

    # Store layer's weight and bias using var_scope
    # Using ResNet
    self.build_model()
    self.saver = tf.train.Saver()


  def block(self, x):
    filters_in = x.shape[0]
  
  def bn(self, x):
    pass

  def fc(self, x):
    pass

  def conv(self, x, params):
    pass

  
  def build_architecture(self):
    filter_size = 3
    strides = [1, 2, 2, 2]
    filters = [64, 128, 256, 512]
    num_blocks_per_resnet = [2, 2, 2, 2]

    # First Convolutional Network
    with tf.variable_scope('init'):
      x = self.X
      output_filters = filters[0]
      x = ops.conv2d(x, output_filters, filter_size, scope='init_conv')

    orig_x = x # Res from previous block
  
    # Using residual blocks...
    for block_num in range(1, 5):
      with tf.variable_scope('unit_{}_0'.format(block_num)):
	activate_before_residual = True if block_num == 1 else False
	x = residual_block(
	  x,
	  filters[block_num - 1],
	  filters[block_num],
	  strides[block_num - 1],
	  activate_before_residual=activate_before_residual)
      for i in range(1, num_blocks_per_resnet[block_num]):
	with tf.variable_scope('unit_{}_{}'.format(block_num, i)):
	  x = residual_block(
	    x,
	    filters[block_num],
	    filters[block_num],
	    1,
	    activate_before_residual=activate_before_residual)
      x, orig_x = _res_add(filters[block_num - 1], filters[block_num],
			    strides[block_num - 1], x, orig_x)  
      
    with tf.variable_scope('unit_last'):
      x = ops.batch_norm(x, scope='final_bn')
      x = tf.nn.relu(x)
      x = ops.global_avg_pool(x)
      self.orig_logits = ops.fc(x, num_classes)
  
  def build_model(self):
    self.X = tf.placeholder(tf.float32, [None, self.num_input], name='x_input')
    self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name='y_input')
    self.soft_Y = tf.placeholder(tf.float32, [None, self.num_classes], name='soft_y')
    self.flag = tf.placeholder(tf.bool, name='flag')

    self.softmax_temperature = tf.placeholder(tf.float32, name='temperature')
    with tf.variable_scope('resnet'):
      self.build_architecture()

    with tf.variable_scope('student_network'):
      self.logits = self.orig_logits / self.softmax_temperature
      self.prob_origin = tf.nn.softmax(self.orig_logits)

      self.correct_pred = tf.equal(tf.argmax(self.prob_origin, 1), tf.argmax(self.Y, 1))
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
      self.op_loss_soft = tf.square(self.softmax_temperature) * \
			  tf.cond(self.flag,
				  true_fn=lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
				    logits=self.logits, labels=self.soft_Y)),
				  false_fn=lambda: 0.0) + tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.orig_logits, labels=self.Y)
      )

      optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
      self.train_op = optimizer.minimize(self.op_loss_soft)

  def train(self, teacher_model=None):
    self.sess.run(tf.global_variables_initializer())

    print("Start Training Student Network")
    data = self.dataset
    max_accuracy = 0

    for step in range(1, self.num_steps + 1):
      x_batch, y_true_batch = data.random_batch()
      soft_targets = np.zeros((self.batch_size, self.num_classes))
      flag = False
      if teacher_model is not None:
	soft_targets = teacher_model.predict(x_batch, self.temperature)
	flag = True
      x_batch = x_batch.reshape(-1, 3 * 32 * 32)
      _ = self.sess.run(self.train_op, feed_dict={
	self.X: x_batch, self.Y: y_true_batch, self.soft_Y: soft_targets, self.softmax_temperature: self.temperature, self.flag: flag
      })
