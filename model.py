import tensorflow as tf
import sys
import os
import numpy as np

sys.path.append('./Utils')
from mnist import MNIST


class TeacherModel(object):
    def __init__(self, sess, num_steps=10000, num_classes=10, dropout_prob=1.0,
                 temperature=5, input_height=28, input_width=28, batch_size=64, learning_rate=0.001, checkpoint_dir="MNIST_Teacher"):
        self.sess = sess
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.num_input = input_height * input_width
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.temperature = temperature
        self.display_step = 100
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = "Teacher"
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file)

        # Store layer's weight and bias using var_scope
        # Using Convolutional Neural Network
        with tf.variable_scope('teacher_weights'):
            self.wc1 = tf.get_variable('wc1', [5, 5, 1, 32], initializer=tf.random_normal_initializer())
            self.wc2 = tf.get_variable('wc2', [5, 5, 32, 64], initializer=tf.random_normal_initializer())
            self.wd1 = tf.get_variable('wd1', [7 * 7 * 64, 1024], initializer=tf.random_normal_initializer())
            self.out = tf.get_variable('out', [1024, self.num_classes], initializer=tf.random_normal_initializer())

        with tf.variable_scope('teacher_bias'):
            self.bc1 = tf.get_variable('bc1', [32], initializer=tf.zeros_initializer())
            self.bc2 = tf.get_variable('bc2', [64], initializer=tf.zeros_initializer())
            self.bd1 = tf.get_variable('bd1', [1024], initializer=tf.zeros_initializer())
            self.bout = tf.get_variable('out', [self.num_classes], initializer=tf.zeros_initializer())

        self.build_model()
        self.saver = tf.train.Saver()

    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.num_input], name='x_input')
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name='y_input')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
        self.softmax_temperature = tf.placeholder(tf.float32, name='temperature')

        with tf.variable_scope('teacher_network'):
            x = tf.reshape(self.X, shape=[-1, 28, 28, 1])
            conv1 = tf.nn.conv2d(x, self.wc1, [1, 1, 1, 1], 'SAME')
            conv1 = tf.nn.bias_add(conv1, self.bc1)
            conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv2 = tf.nn.conv2d(conv1, self.wc2, [1, 1, 1, 1], 'SAME')
            conv2 = tf.nn.bias_add(conv2, self.bc2)
            conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv2 = tf.reshape(conv2, shape=[-1, 7 * 7 * 64])
            fc1 = tf.nn.relu(tf.matmul(conv2, self.wd1) + self.bd1)
            fc1 = tf.nn.dropout(fc1, self.keep_prob)
            fc2 = tf.matmul(fc1, self.out) + self.bout

            self.logits = fc2 / self.softmax_temperature
            self.prob_temperature = tf.nn.softmax(fc2 / self.softmax_temperature)
            self.prob_origin = tf.nn.softmax(fc2)

            self.correct_pred = tf.equal(tf.argmax(self.prob_temperature, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            self.op_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits, labels=self.Y
            ))

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.op_loss)

        # tf.summary.scalar("loss", self.op_loss)
        # tf.summary.scalar("accuracy", self.accuracy)
        #
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.name, var)
        #
        # def merge_function(scope_str):
        #     from tensorflow.python.framework import ops as _ops
        #     key = _ops.GraphKeys.SUMMARIES
        #     summary_ops = _ops.get_collection(key, scope=scope_str)
        #     if not summary_ops:
        #         return None
        #     else:
        #         return tf.summary.merge(summary_ops)
        #
        # self.merged_summary_op = merge_function(self.)

    def train(self):
        self.sess.run(tf.global_variables_initializer())

        print("Start Training Teacher Network")
        data = MNIST(data_dir="data/MNIST/")
        max_accuracy = 0

        for step in range(1, self.num_steps + 1):
            x_batch, y_true_batch, _ = data.random_batch(batch_size=self.batch_size)
            acc, loss, _ = self.sess.run([self.accuracy, self.op_loss, self.train_op], feed_dict={
                self.X: x_batch, self.Y: y_true_batch, self.keep_prob: self.dropout_prob, self.softmax_temperature: 1.0
            })

            if step % self.display_step == 0:
                x_val = data.x_val
                y_val = data.y_val
                acc, loss = self.sess.run([self.accuracy, self.op_loss], feed_dict={
                    self.X: x_val, self.Y: y_val, self.keep_prob: 1.0, self.softmax_temperature: 1.0
                })
                print("Step " + str(step) + ", Validation Loss= " + "{:.4f}".format(
                    loss) + ", Validation Accuracy= " + "{:.3f}".format(acc)
                )

                if acc > max_accuracy:
                    save_path = self.saver.save(self.sess, self.checkpoint_path)
                    print("Model Checkpointed to %s " % save_path)
                    max_accuracy = acc

    def predict(self, x_batch, temp):
        logits = self.sess.run(self.prob_origin, feed_dict={self.X: x_batch, self.softmax_temperature: temp, self.keep_prob: 1.0})

        return logits

    def load_model_from_file(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())

class StudentModel(object):
    def __init__(self, sess, num_steps=10000, num_classes=10,
                 temperature=1.0, input_height=28, input_width=28, batch_size=64, learning_rate=0.001,
                 checkpoint_dir="MNIST_Student", checkpoint_file="Student"):
        self.sess = sess
        self.num_hidden1 = 32
        self.num_hidden2 = 32
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.num_input = input_height * input_width
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.temperature = temperature
        self.display_step = 100
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_file
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file + ".cpkt")

        # Store layer's weight and bias using var_scope
        # Using Convolutional Neural Network
        with tf.variable_scope('student_weights'):
            self.hw1 = tf.get_variable('hw1', [self.num_input, self.num_hidden1], initializer=tf.random_normal_initializer())
            self.hw2 = tf.get_variable('hw2', [self.num_hidden1, self.num_hidden2], initializer=tf.random_normal_initializer())
            self.out = tf.get_variable('out', [self.num_hidden2, self.num_classes], initializer=tf.random_normal_initializer())

        with tf.variable_scope('student_bias'):
            self.hb1 = tf.get_variable('hb1', [self.num_hidden1], initializer=tf.zeros_initializer())
            self.hb2 = tf.get_variable('hb2', [self.num_hidden2], initializer=tf.zeros_initializer())
            self.bout = tf.get_variable('out', [self.num_classes], initializer=tf.zeros_initializer())

        self.build_model()
        self.saver = tf.train.Saver()

    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.num_input], name='x_input')
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name='y_input')
        self.soft_Y = tf.placeholder(tf.float32, [None, self.num_classes], name='soft_y')
        self.flag = tf.placeholder(tf.bool, name='flag')

        self.softmax_temperature = tf.placeholder(tf.float32, name='temperature')

        with tf.variable_scope('student_network'):
            fc1 = tf.matmul(self.X, self.hw1) + self.hb1
            fc1 = tf.nn.relu(fc1)
            fc2 = tf.matmul(fc1, self.hw2) + self.hb2
            fc2 = tf.nn.relu(fc2)
            fc3 = tf.matmul(fc2, self.out) + self.bout

            self.logits = fc3 / self.softmax_temperature
            self.prob_origin = tf.nn.softmax(fc3)

            self.correct_pred = tf.equal(tf.argmax(self.prob_origin, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            self.op_loss_soft = tf.square(self.softmax_temperature) * \
                                tf.cond(self.flag,
                                        true_fn=lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                                                                        logits=self.logits, labels=self.soft_Y)),
                                        false_fn=lambda: 0.0) + tf.reduce_mean(
                                            tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=self.Y)
                                    )

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.op_loss_soft)

    def train(self, teacher_model):
        self.sess.run(tf.global_variables_initializer())

        print("Start Training Student Network")
        data = MNIST(data_dir="data/MNIST/")
        max_accuracy = 0

        for step in range(1, self.num_steps + 1):
            x_batch, y_true_batch, _ = data.random_batch(batch_size=self.batch_size)
            soft_targets = np.zeros((self.batch_size, self.num_classes))
            flag = False
            if teacher_model is not None:
                soft_targets = teacher_model.predict(x_batch, self.temperature)
                flag = True

            _ = self.sess.run(self.train_op, feed_dict={
                self.X: x_batch, self.Y: y_true_batch, self.soft_Y: soft_targets, self.softmax_temperature: self.temperature, self.flag: flag
            })

            if step % self.display_step == 0:
                x_val = data.x_val
                y_val = data.y_val
                acc, loss = self.sess.run([self.accuracy, self.op_loss_soft], feed_dict={
                    self.X: x_val, self.Y: y_val, self.soft_Y: soft_targets, self.softmax_temperature: 1.0, self.flag: False
                })
                print("Step " + str(step) + ", Validation Loss= " + "{:.4f}".format(
                    loss) + ", Validation Accuracy= " + "{:.3f}".format(acc)
                )

                if acc > max_accuracy:
                    save_path = self.saver.save(self.sess, self.checkpoint_path)
                    print("Model Checkpointed to %s " % save_path)
                    max_accuracy = acc
