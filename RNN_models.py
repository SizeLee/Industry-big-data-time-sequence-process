import numpy as np
import tensorflow as tf
import json
import random

class FixedLengthRNN:
    def __init__(self, fixed_length, input_size, class_num, cell_type='lstm', foresight_steps=0,
                 network_hyperparameters='./data/network_hyperparameters.json'):
        self.fixed_length = fixed_length
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.input_size = input_size
        self.cell_type = cell_type  ## lstm, gru, sru
        self.dtype = tf.float32
        self.class_num = class_num
        # self.parameters = []
        self.foresight_steps = foresight_steps
        with open(network_hyperparameters, 'r') as f:
            self.network_hyperparameter = json.load(f)
        with self.graph.as_default():
            self._build_RNN()
            self.initializer = tf.global_variables_initializer()
            self.sess.run(self.initializer)
        return

    def _build_RNN(self):
        with tf.variable_scope('input'):
            self.input = tf.placeholder(shape=[None, self.fixed_length, self.input_size],
                                        dtype=self.dtype, name='input_sequence')
        with tf.name_scope('RNN_of_%s' % self.cell_type):
            rnn_layer_num = self.network_hyperparameter['rnn_layer_num']
            layer_input = self.input
            layer_out = layer_input
            for i in range(rnn_layer_num):
                # print(layer_input.shape)
                layer_out = self._rnn_layer(i+1, layer_input)
                layer_input = layer_out

        with tf.name_scope('output_layer'):
            last_out = tf.reshape(layer_out[:, -1, :], [-1, int(layer_out.shape[2])])
            # print(last_out.shape)
            fc_layer_num = self.network_hyperparameter['fc_layer_num']
            out_put = last_out
            linear_out = out_put
            for i in range(fc_layer_num):
                out_put, linear_out = self._dense_layer('fc%d' % (i+1), out_put,
                                            self.network_hyperparameter['fc_layers']['fc%d' % (i+1)]['size'],
                                            self.network_hyperparameter['fc_layers']['fc%d' % (i+1)]['activation_func'])
            # out_put, linear_out = self._dense_layer('fc_out', out_put, self.class_num, 'softmax')
            self.predict = tf.argmax(out_put, axis=1)

        with tf.name_scope('training_and_judging'):
            self.y = tf.placeholder(shape=[None, self.class_num], dtype=self.dtype, name='labels')
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=linear_out, labels=self.y))
            optimizer = tf.train.AdamOptimizer() # learning rate could be adjust
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

    def _rnn_layer(self, layer_id, layer_input):
        with tf.name_scope('layer%d' % layer_id):
            with tf.variable_scope('layer%d_%s_cell' % (layer_id, self.cell_type)):
                if self.cell_type.lower() == 'lstm':
                    cell = tf.nn.rnn_cell.LSTMCell(self.network_hyperparameter['rnn_layers']['layer%d' % layer_id]['hidden_size'], name='cell')
                elif self.cell_type.lower() == 'gru':
                    cell = tf.nn.rnn_cell.GRUCell(self.network_hyperparameter['rnn_layers']['layer%d' % layer_id]['hidden_size'], name='cell')
                elif self.cell_type.lower() == 'sru':
                    cell = tf.contrib.rnn.SRUCell(self.network_hyperparameter['rnn_layers']['layer%d' % layer_id]['hidden_size'], name='cell')
                else:
                    raise Exception('error cell type')
                with tf.name_scope('initial_state'):
                    initial_state = cell.zero_state(tf.shape(layer_input)[0], dtype=self.dtype)
                # c, h = initial_state
                # print(c.shape)
                # print(h.shape)

                out = []
                cell_state = initial_state
                for t in range(self.fixed_length):
                    if t > 0:
                        tf.get_variable_scope().reuse_variables()
                    cell_out, cell_state = cell(layer_input[:, t, :], cell_state)
                    out.append(tf.reshape(cell_out, [-1, 1, cell_out.shape[1]]))
                # print(out[0].shape)
                out = tf.concat(out, axis=1)

        return out

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

    def train(self, data, labels, epoches, batch_size, train_set_sample_ids, foresight_steps=None, reset_flag=False):
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
            data_set = self._data_generator(data, labels, self.fixed_length, batch_size, train_set_sample_ids)
            for batch_data, batch_label in data_set:
                loss, _ = self.sess.run([self.loss, self.train_step],
                                        feed_dict={self.input: batch_data, self.y: batch_label})
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
        data_set = self._data_generator(data, labels, self.fixed_length, batch_size, sample_ids)
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
            yield batch_data, batch_label

        return  # 'one epoch done'
