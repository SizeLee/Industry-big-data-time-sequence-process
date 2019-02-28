import tensorflow as tf
import numpy as np
import json
import random

class BPNN:
    def __init__(self, fixed_length, input_size, class_num, foresight_steps=0,
                 network_hyperparameters='./data/bpnn_network_hyperparameters.json'):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.sequence_length = fixed_length
        self.input_size = input_size
        self.class_num = class_num
        self.foresight_steps = foresight_steps
        self.dtype = tf.float32

        with open(network_hyperparameters, 'r') as f:
            self.network_hyperparameter = json.load(f)

        with self.graph.as_default():
            self._build_NN()
            self.initializer = tf.global_variables_initializer()
            self.sess.run(self.initializer)

        return

    def _build_NN(self):
        with tf.variable_scope('input'):
            self.input = tf.placeholder(shape=[None, self.sequence_length, self.input_size],
                                        dtype=self.dtype, name='input_sequence')
                                        # shape = [sample, sequence_length, feature_num]

        layer_input = self.input

        with tf.name_scope('fc_layers'):
            fc_layers_num = self.network_hyperparameter['fc_layers_num']
            last_out = tf.reshape(layer_input, [-1, layer_input.shape[1]*layer_input.shape[2]])
            out_put = last_out
            linear_out = out_put
            for i in range(fc_layers_num):
                out_put, linear_out = self._dense_layer('fc%d' % (i + 1), out_put,
                                    self.network_hyperparameter['fc_layers']['fc%d' % (i + 1)]['size'],
                                    self.network_hyperparameter['fc_layers']['fc%d' % (i + 1)]['activation_func'])
            self.predict = tf.argmax(out_put, axis=1)

        with tf.name_scope('training_and_judging'):
            self.y = tf.placeholder(shape=[None, self.class_num], dtype=self.dtype, name='labels')
            self.weight_matrix = tf.placeholder(shape=[None, 1], dtype=self.dtype, name='weight_matrix')
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=linear_out, labels=self.y)
                                      * tf.reshape(self.weight_matrix, [-1]))
            self.learning_rate = tf.placeholder(dtype=self.dtype, name='learning_rate')
            optimizer = tf.train.AdamOptimizer(self.learning_rate) # learning rate could be adjust
            self.train_step = optimizer.minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict, tf.argmax(self.y, axis=1)), dtype=self.dtype))

        return

    def save_model(self, save_dir, global_step=None):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(sess=self.sess, save_path=save_dir+'/model.ckpt', global_step=global_step)
        return

    def load_model(self, load_dir):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, load_dir+'/model.ckpt')
        return

    def _dense_layer(self, layer_name, layer_input, layer_out_size, activation_func='relu'):
        with tf.name_scope(layer_name):
            with tf.variable_scope(layer_name+'_variables'):
                # w = tf.Variable(initial_value=tf.truncated_normal([layer_input.shape[1], layer_out_size], stddev=1e-1,
                #                                                   dtype=self.dtype), name='weight')
                w = tf.get_variable('weight', shape=[layer_input.shape[1], layer_out_size], dtype=self.dtype,
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(initial_value=tf.constant(0, shape=[layer_out_size], dtype=self.dtype), name='bias')
                linear_out = tf.nn.bias_add(tf.matmul(layer_input, w), b)
                out = linear_out
                if activation_func.lower() == 'relu':
                    out = tf.nn.relu(out)
                elif activation_func.lower() == 'sigmoid':
                    out = tf.nn.sigmoid(out)
                elif activation_func.lower() == 'tanh':
                    out = tf.nn.tanh(out)
                elif activation_func.lower() == 'softmax':
                    out = tf.nn.softmax(out)
                elif activation_func.lower() == 'linear':
                    out = out
                # illegal input string will be treat as linear
        return out, linear_out

    def train(self, data, labels, epoches, batch_size, train_set_sample_ids, learning_rate=0.001,
              foresight_steps=None, reset_flag=False):
        if reset_flag:
            self.sess.run(self.initializer)
        if foresight_steps is not None:
            try:
                self.foresight_steps = int(foresight_steps)
            except:
                print('Wrong format of value of variable foresight_steps')
                pass

        for i in range(epoches):
            print('epoch%d:' % i)
            data_set = self._data_generator(data, labels, self.sequence_length, batch_size, train_set_sample_ids)
            for batch_data, batch_label, weight in data_set:
                loss, _ = self.sess.run([self.loss, self.train_step],
                                        feed_dict={self.input: batch_data, self.y: batch_label,
                                                   self.learning_rate: learning_rate,
                                                   self.weight_matrix: weight})
                print(loss)
            print()
            # self.sess.run(self.accuracy, feed_dict={})
        accuracy = self._cal_accuracy(data, labels, batch_size, train_set_sample_ids)
        print('accuracy on training set: %f' % accuracy)
        return

    def test(self, data, label, test_set_sample_ids=None, batch_size=1024, data_set_name='test set'):
        accuracy = self._cal_accuracy(data, label, batch_size, test_set_sample_ids)
        print('accuracy on %s: %f' % (data_set_name, accuracy))
        return

    def _cal_accuracy(self, data, labels, batch_size, sample_ids=None):
        data_set = self._data_generator(data, labels, self.sequence_length, batch_size, sample_ids)
        batch_count = 0
        accuracy = 0.
        for batch_data, batch_label in data_set:
            batch_count += batch_label.shape[0]
            accuracy += batch_label.shape[0] * self.sess.run(self.accuracy,
                                                             feed_dict={self.input: batch_data, self.y: batch_label})
        accuracy /= batch_count
        return accuracy

    def _data_generator(self, data, labels, length, batch_size, sample_ids=None):
        if sample_ids is None:
            sample_ids = set([i for i in range(data.shape[0]-length+1-self.foresight_steps)])
        else:
            sample_ids = set(filter(lambda x: x < (data.shape[0]-length+1-self.foresight_steps), sample_ids))

        while len(sample_ids) > 0:
            if batch_size >= len(sample_ids):
                batch = list(sample_ids)
            else:
                batch = random.sample(sample_ids, batch_size)
            sample_ids = sample_ids - set(batch)
            batch_data = np.zeros((batch_size, length, data.shape[1]))
            batch_label = np.zeros((batch_size,  self.class_num))
            for i, each_sample in enumerate(batch):
                batch_data[i, :, :] = data[each_sample:(each_sample+length), :]
                batch_label[i, int(labels[each_sample + length - 1 + self.foresight_steps, 0])] = 1
            # weight = np.mean(batch_label, axis=0).dot(np.array([[1, 1, 1, 10]]).T)
            weight = batch_label.dot(np.array([[1, 1, 1, 5000]]).T)
            yield batch_data, batch_label, weight

        return  # 'one epoch done'


if __name__ == '__main__':
    t = BPNN(30, 2, 4, 3)
