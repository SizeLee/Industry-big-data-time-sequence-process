import tensorflow as tf
import numpy as np
import math
import configparser
import json
import random
import time

class HMM_supervised:
    def __init__(self, sequence_length, feature_size, class_num, foresight_steps):
        self.sequence_length = sequence_length
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.feature_size = feature_size
        self.dtype = tf.float64
        self.class_num = class_num
        # self.parameters = []
        self.foresight_steps = foresight_steps

        with self.graph.as_default():
            self._build_markov_chain()
            self.initializer = tf.global_variables_initializer()

        self.sess.run(self.initializer)

        return

    def _build_markov_chain(self):
        with tf.variable_scope('input_one_sample'):
            self.one_sample = tf.placeholder(dtype=self.dtype, shape=[self.sequence_length, self.feature_size])
        with tf.variable_scope('Gaussian_distribute_parameters'):
            miu = tf.get_variable('miu', shape=[1, self.class_num ** 2, self.feature_size],
                                  initializer=tf.random_normal_initializer(), dtype=self.dtype)
            sigma = tf.get_variable('sigma', shape=[1, self.class_num ** 2, self.feature_size],
                                    initializer=tf.ones_initializer(), dtype=self.dtype)

            sigma = 1/math.sqrt(2 * math.pi) * sigma

            one_sample = tf.expand_dims(self.one_sample, axis=1)
            exp = tf.exp(-tf.reduce_sum(((one_sample - miu) ** 2) / (2 * (sigma ** 2)), axis=2)/tf.Variable(100, dtype=self.dtype))
            factor = 1/(tf.reduce_prod(math.sqrt(2 * math.pi) * sigma, axis=2))
            b_prob = tf.reshape(factor * exp, [self.sequence_length, self.class_num, self.class_num], name='b_prob')
            # [sample_length, class_num, ->class_num]
            # self.check = one_sample
            # self.check = sigma
            # self.check = (one_sample - miu) ** 2
            # self.check = ((one_sample - miu) ** 2) / (2 * (sigma ** 2))
            # self.check = -tf.reduce_sum(((one_sample - miu) ** 2) / (2 * (sigma ** 2)), axis=2)
            # self.check = exp
            # self.check = factor
            # self.check = b_prob
            # print(self.check.shape)


        with tf.variable_scope('a_transfer_prob'):
            a_transfer_prob = tf.get_variable('a', initializer=tf.ones(shape=[self.sequence_length,
                                                                              self.class_num,
                                                                              self.class_num], dtype=self.dtype)
                                                               /self.class_num, dtype=self.dtype)
            a_transfer_prob = tf.nn.softmax(a_transfer_prob, axis=2)  # if gradient disappears, try share a_prob on different time steps

        with tf.name_scope('all_steps_transfer_prob'):
            all_step_transfer_prob = a_transfer_prob * b_prob

        with tf.variable_scope('init_status'):
            init_status_prob = tf.get_variable('init_status_prob',
                                               initializer=tf.ones(shape=[self.class_num, 1], dtype=self.dtype)/self.class_num)
        with tf.name_scope('markov_chain'):
            predict_logit, pre = self._markov_chain(all_step_transfer_prob, init_status_prob)
            self.predict = tf.argmax(predict_logit)

        with tf.name_scope('train'):
            self.y = tf.placeholder(self.dtype, shape=[class_num], name='real_label')
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=predict_logit, labels=self.y)
            optimizer = tf.train.AdamOptimizer()
            self.train_step = optimizer.minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict, tf.argmax(self.y)), dtype=self.dtype))

            # self.check = tf.reduce_max(tf.gradients(self.loss, miu))
            # self.check = self.loss
        return

    def _markov_chain(self, all_step_transfer_prob, init_prob):
        last_pro_on_status = init_prob
        for i in range(self.sequence_length):
            pre = last_pro_on_status
            last_pro_on_status = tf.reduce_max(last_pro_on_status * all_step_transfer_prob[i, :, :] * tf.Variable(20, dtype=self.dtype), axis=0)
            last_pro_on_status = tf.expand_dims(last_pro_on_status, axis=1)
        last_pro_on_status = tf.reshape(last_pro_on_status, [-1])
        return last_pro_on_status, pre  # shape is [class_num]

    def train_v2(self, data, labels, samples_length, epoches, train_set_sample_ids, learning_rate=0.001,
                 foresight_steps=None, reset_flag=False):
        batch_size = 1
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
            data_set = self._data_generator_v2(data, labels, self.sequence_length, samples_length, batch_size,
                                               train_set_sample_ids)
            for batch_data, batch_label in data_set:
                # print(weight)
                batch_data = batch_data.reshape([batch_data.shape[1], batch_data.shape[2]])
                batch_label = batch_label.reshape([batch_label.shape[1]])
                # print(batch_data)
                # print(batch_label)
                loss, _ = self.sess.run([self.loss, self.train_step],
                                        feed_dict={self.one_sample: batch_data, self.y: batch_label})
                # loss, _, check = self.sess.run([self.loss, self.train_step, self.check],
                #                                feed_dict={self.one_sample: batch_data, self.y: batch_label})
                # print(check)
                print(loss)

            print()
            # self.sess.run(self.accuracy, feed_dict={})
        accuracy = self._cal_accuracy_v2(data, labels, samples_length, batch_size, train_set_sample_ids)
        print('accuracy on training set: %f' % accuracy)
        return

    def test_v2(self, data, label, samples_length, test_set_sample_ids=None, data_set_name='test set'):
        batch_size = 1
        accuracy = self._cal_accuracy_v2(data, label, samples_length, batch_size, test_set_sample_ids)
        print('accuracy on %s: %f' % (data_set_name, accuracy))
        return

    def _cal_accuracy_v2(self, data, labels, samples_length, batch_size=1, sample_ids=None):
        data_set = self._data_generator_v2(data, labels, self.sequence_length, samples_length, batch_size, sample_ids)
        batch_count = 0
        accuracy = 0.
        for batch_data, batch_label in data_set:
            batch_count += batch_size
            batch_data = batch_data.reshape([batch_data.shape[1], batch_data.shape[2]])
            batch_label = batch_label.reshape([batch_label.shape[1]])
            accuracy += batch_size * self.sess.run(self.accuracy,
                                                   feed_dict={self.one_sample: batch_data, self.y: batch_label})
        accuracy /= batch_count
        return accuracy

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
            batch_label = np.zeros((batch_size, self.class_num))
            for i, each_sample in enumerate(batch):
                batch_data[i, :int(samples_length[each_sample, 0]), :] \
                    = data[each_sample:(each_sample + int(samples_length[each_sample, 0])), :]

                batch_label[i, int(labels[each_sample, 0])] = 1

            yield batch_data, batch_label

        return  # 'one epoch done'


if __name__ == '__main__':
    common_para = configparser.ConfigParser()
    common_para.read('common_para.ini')
    sequence_fix_length = common_para['common_parameters'].getint('sequence_fix_length')
    foresight_steps = common_para['common_parameters'].getint('foresight_steps')
    class_num = common_para['common_parameters'].getint('class_num')

    data = np.load(common_para['path']['data_file'])
    # print(data['features'].shape)
    # print(data['labels'])
    # print(data['samples_length'])

    mean_on_feature = np.mean(data['features'], axis=0, keepdims=True)
    min_on_feature = np.min(data['features'], axis=0, keepdims=True)
    max_on_feature = np.max(data['features'], axis=0, keepdims=True)

    ### generate training samples' id

    with open(common_para['path']['data_set_ids_file'], 'r') as f:
        data_set_ids = json.load(f)
    training_sample_ids = data_set_ids['training_set']
    test_sample_ids = data_set_ids['test_set']
    print(len(training_sample_ids))
    print(len(test_sample_ids))

    # print(max_on_feature-min_on_feature)
    data_features = (data['features']-mean_on_feature)/((max_on_feature - min_on_feature)+0.1)

    hmm = HMM_supervised(sequence_fix_length, data['features'].shape[1], class_num, foresight_steps)
    start =time.time()
    hmm.train_v2(data_features, data['labels'], data['samples_length'], 1, training_sample_ids,
                  foresight_steps=0, reset_flag=True)
    end = time.time()
    print(end-start)
    hmm.test_v2(data_features, data['labels'], data['samples_length'], test_sample_ids)

