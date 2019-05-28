import tensorflow as tf
import numpy as np
import json
import random
import time


class ConvSequence2One:
    def __init__(self, fixed_length, input_size, class_num, foresight_steps=0,
                 network_hyperparameters='./data/cnn_network_hyperparameters.json'):
        self.graph = tf.Graph()
        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2
        self.sess = tf.Session(graph=self.graph, config=tf_config)
        self.sequence_length = fixed_length
        self.input_size = input_size
        self.class_num = class_num
        self.foresight_steps = foresight_steps
        self.dtype = tf.float32

        with open(network_hyperparameters, 'r') as f:
            self.network_hyperparameter = json.load(f)

        with self.graph.as_default():
            self._build_CNN()
            self._build_summary_node()
            self.initializer = tf.global_variables_initializer()
            self.sess.run(self.initializer)

        return

    def _build_CNN(self):
        with tf.variable_scope('input'):
            self.input = tf.placeholder(shape=[None, self.sequence_length, self.input_size],
                                        dtype=self.dtype, name='input_sequence')
                                        # shape = [sample, sequence_length, feature_num]

        layer_input = self.input

        with tf.name_scope('cnn_layers'):
            conv_layers_num = self.network_hyperparameter['cnn_layers_num']
            for i in range(conv_layers_num):
                layer_input = self._cnn_layer(i+1, layer_input)

        with tf.name_scope('output_layers'):
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
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=linear_out, labels=self.y)
                                      * tf.reshape(self.weight_matrix, [-1]))
            self.learning_rate = tf.placeholder(dtype=self.dtype, name='learning_rate')
            optimizer = tf.train.AdamOptimizer(self.learning_rate) # learning rate could be adjust
            self.train_step = optimizer.minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict, tf.argmax(self.y, axis=1)), dtype=self.dtype))

            self.batch_loss_summary = tf.summary.scalar('batch_loss', self.loss)
            self.batch_accuracy_summary = tf.summary.scalar('batch_accuracy', self.accuracy)
            self.batch_summary = tf.summary.merge([self.batch_loss_summary, self.batch_accuracy_summary])

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

    def _build_summary_node(self):
        with tf.name_scope('summary_node'):
            self.whole_loss_node = tf.placeholder(tf.float32)
            self.whole_loss_summary = tf.summary.scalar('whole_loss', self.whole_loss_node)
            self.whole_accuracy_node = tf.placeholder(tf.float32)
            self.whole_accuracy_summary = tf.summary.scalar('whole_accuracy', self.whole_accuracy_node)
            self.whole_summary = tf.summary.merge([self.whole_loss_summary, self.whole_accuracy_summary])

    def _cnn_layer(self, layer_id, layer_input):
        kernel_size = self.network_hyperparameter['cnn_layers']['layer_%d' % layer_id]['kernel_size']
        kernels_num = self.network_hyperparameter['cnn_layers']['layer_%d' % layer_id]['kernels_num']
        stride = self.network_hyperparameter['cnn_layers']['layer_%d' % layer_id]['stride']
        padding_pattern = self.network_hyperparameter['cnn_layers']['layer_%d' % layer_id]['padding']
        activation_pattern = self.network_hyperparameter['cnn_layers']['layer_%d' % layer_id]['activation'].lower()
        with tf.name_scope('layer_%d' % layer_id):
            with tf.variable_scope('cnn_blocks_of_layer_%d' % layer_id):
                kernel = tf.get_variable('kernel', shape=[kernel_size, layer_input.shape[-1], kernels_num],
                                         dtype=self.dtype,
                                         initializer=tf.contrib.layers.xavier_initializer())
                # print(kernel.name)
                conv = tf.nn.conv1d(layer_input, kernel, stride=stride, padding=padding_pattern, name='conv')
                bias = tf.get_variable('bias', shape=[kernels_num], dtype=self.dtype,
                                       initializer=tf.zeros_initializer())
                v = tf.nn.bias_add(conv, bias)

                if activation_pattern == 'glu':
                    out = v[:, :, :kernels_num//2] * tf.nn.sigmoid(v[:, :, kernels_num//2:])
                elif activation_pattern == 'relu':
                    out = tf.nn.relu(v)
                elif activation_pattern == 'sigmoid':
                    out = tf.nn.sigmoid(v)
                elif activation_pattern == 'tanh':
                    out = tf.nn.tanh(v)
                elif activation_pattern == 'softplus':
                    out = tf.nn.softplus(v)
                else:
                    out = v
                # illegal activation pattern string means linear activation

        return out

    def _dense_layer(self, layer_name, layer_input, layer_out_size, activation_func='relu'):
        with tf.name_scope(layer_name):
            with tf.variable_scope(layer_name+'_variables'):
                # w = tf.Variable(initial_value=tf.truncated_normal([layer_input.shape[1], layer_out_size], stddev=1e-1,
                #                                                   dtype=self.dtype), name='weight')
                w = tf.get_variable('weight', shape=[layer_input.shape[-1], layer_out_size], dtype=self.dtype,
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

    def _dropout(self, layer_input, dropout_rate, is_training):
        return tf.layers.dropout(layer_input, dropout_rate, training=is_training)

    def _batch_norm(self, layer_input, is_training):
        return tf.layers.batch_normalization(layer_input, training=is_training)

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
        for batch_data, batch_label, _ in data_set:
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
            weight = batch_label.dot(np.array([[1, 1, 1, 1]]).T)
            yield batch_data, batch_label, weight

        return  # 'one epoch done'

    def train_v2(self, data, labels, samples_length, epoches, batch_size, train_set_sample_ids, test_set_ids,
                 learning_rate=0.001, foresight_steps=None, reset_flag=False, record_flag=True,
                 log_dir='./data/log/cnn_models', random_seed=None):

        if random_seed is not None:
            random.seed(random_seed)

        if reset_flag:
            self.sess.run(self.initializer)
        if foresight_steps is not None:
            try:
                self.foresight_steps = int(foresight_steps)
            except:
                print('Wrong format of value of variable foresight_steps')
                pass

        if record_flag:
            train_writer = tf.summary.FileWriter(log_dir + '/cnn/train', self.sess.graph)
            test_writer = tf.summary.FileWriter(log_dir + '/cnn/test')

        step_count = 0
        b_losses = []
        b_aces = []
        val_aces = []
        for i in range(epoches):
            print('epoch%d:' % i)
            data_set = self._data_generator_v2(data, labels, self.sequence_length, samples_length, batch_size,
                                               train_set_sample_ids)
            for batch_data, batch_label, weight in data_set:
                # print(weight)
                loss, b_ac, _, batch_summary = self.sess.run([self.loss, self.accuracy, self.train_step, self.batch_summary],
                                                       feed_dict={self.input: batch_data, self.y: batch_label,
                                                                  self.learning_rate: learning_rate,
                                                                  self.weight_matrix: weight})
                # print('step%d: %f' % (step_count, loss))
                b_losses.append(float(loss))
                b_aces.append(float(b_ac))
                if step_count % 10 == 0:
                    val_ac, val_loss = self._cal_accuracy_and_loss_v2(data, labels, samples_length, batch_size,
                                                                      test_set_ids)
                    val_aces.append(float(val_ac))

                # print(check)
                if record_flag:
                    train_writer.add_summary(batch_summary, global_step=step_count)

                if step_count % 100 == 0 and record_flag:
                    self._whole_summary_write(train_writer, step_count, data, labels, samples_length, batch_size,
                                              train_set_sample_ids)

                    self._whole_summary_write(test_writer, step_count, data, labels, samples_length, batch_size,
                                              test_set_ids)

                step_count += 1
            print()
            # self.sess.run(self.accuracy, feed_dict={})

        cnn_info = {'batch_loss': b_losses, 'batch_accuracy': b_aces, 'val_accuracy': val_aces}
        with open('./data/info/cnn_info.json', 'w') as f:
            json.dump(cnn_info, f)

        if record_flag:
            accuracy = self._whole_summary_write(train_writer, step_count, data, labels, samples_length, batch_size,
                                             train_set_sample_ids)

            self._whole_summary_write(test_writer, step_count, data, labels, samples_length, batch_size, test_set_ids)
        else:
            accuracy, _ = self._cal_accuracy_and_loss_v2(data, labels, samples_length, batch_size, train_set_sample_ids)

        print('accuracy on training set: %f' % accuracy)
        if record_flag:
            train_writer.close()
            test_writer.close()
        return

    def _whole_summary_write(self, writer, step, data, labels, samples_length, batch_size, data_set_sample_ids):
        w_accuracy, w_loss = self._cal_accuracy_and_loss_v2(data, labels, samples_length, batch_size,
                                                            data_set_sample_ids)
        whole_summary = self.sess.run(self.whole_summary, feed_dict={self.whole_accuracy_node: w_accuracy,
                                                                     self.whole_loss_node: w_loss})
        writer.add_summary(whole_summary, global_step=step)
        return w_accuracy

    def test_v2(self, data, label, samples_length, test_set_sample_ids=None, batch_size=1024, data_set_name='test set'):
        accuracy = self._cal_accuracy_v2(data, label, samples_length, batch_size, test_set_sample_ids)
        print('accuracy on %s: %f' % (data_set_name, accuracy))
        return

    def _cal_accuracy_v2(self, data, labels, samples_length, batch_size, sample_ids=None):
        data_set = self._data_generator_v2(data, labels, self.sequence_length, samples_length, batch_size, sample_ids)
        batch_count = 0
        accuracy = 0.
        for batch_data, batch_label, _ in data_set:
            batch_count += batch_label.shape[0]
            accuracy += batch_label.shape[0] * self.sess.run(self.accuracy,
                                                             feed_dict={self.input: batch_data, self.y: batch_label})
        accuracy /= batch_count
        return accuracy

    def _cal_accuracy_and_loss_v2(self, data, labels, samples_length, batch_size, sample_ids=None):
        data_set = self._data_generator_v2(data, labels, self.sequence_length, samples_length, batch_size, sample_ids)
        sample_count = 0
        accuracy = 0.
        loss = 0.
        for batch_data, batch_label, weight in data_set:
            sample_count += batch_label.shape[0]
            b_ac, b_loss = self.sess.run([self.accuracy, self.loss],
                                         feed_dict={self.input: batch_data, self.y: batch_label,
                                                    self.weight_matrix: weight})
            accuracy += batch_label.shape[0] * b_ac
            loss += batch_label.shape[0] * b_loss

        accuracy /= sample_count
        loss /= sample_count
        return accuracy, loss

    def _data_generator_v2(self, data, labels, max_length, samples_length, batch_size, sample_ids=None):
        if sample_ids is None:
            sample_ids = set([i for i in range(data.shape[0] - max_length + 1 - self.foresight_steps)])
        else:
            sample_ids = set(filter(lambda x: x < (data.shape[0] - max_length + 1 - self.foresight_steps), sample_ids))

        while len(sample_ids) > 0:
            if batch_size >= len(sample_ids):
                batch = list(sample_ids)
            else:
                batch = random.sample(sample_ids, batch_size)
            sample_ids = sample_ids - set(batch)
            batch_data = np.zeros((batch_size, max_length, data.shape[1]))
            batch_label = np.zeros((batch_size,  self.class_num))
            for i, each_sample in enumerate(batch):
                batch_data[i, :int(samples_length[each_sample, 0]), :] \
                    = data[each_sample:(each_sample + int(samples_length[each_sample, 0])), :]

                batch_label[i, int(labels[each_sample, 0])] = 1
            # weight = np.mean(batch_label, axis=0).dot(np.array([[1, 1, 1, 10]]).T)
            weight = batch_label.dot(np.ones((self.class_num, 1)))
            yield batch_data, batch_label, weight

        return  # 'one epoch done'

    def test_time(self, data, labels, sample_length, test_set_ids, save_dir='./data/confusion_matrix/cnn'):
        whole_time, y_true, y_predict = self._get_test_result_and_running_time(data, labels, sample_length, 1024,
                                                                               test_set_ids)
        np.savez(save_dir + '/cnn_predict.npz', y_predict=y_predict, y_true=y_true)
        return whole_time

    def _get_test_result_and_running_time(self, data, labels, samples_length, batch_size, sample_ids=None):
        data_set = self._data_generator_v2(data, labels, self.sequence_length, samples_length, batch_size, sample_ids)
        y_true = []
        y_predict = []
        whole_time = 0.
        for batch_data, batch_label, _ in data_set:
            b_true = np.argmax(batch_label, axis=1)
            start = time.time()
            b_pre = self.sess.run(self.predict, feed_dict={self.input: batch_data, self.y: batch_label})
            b_pre = b_pre.reshape([-1])
            end = time.time()
            whole_time += end - start
            y_true.append(b_true)
            y_predict.append(b_pre)

        y_true = np.concatenate(y_true)
        y_predict = np.concatenate(y_predict)

        return whole_time, y_true, y_predict


if __name__ == '__main__':
    t = ConvSequence2One(30, 2, 4, 3)
