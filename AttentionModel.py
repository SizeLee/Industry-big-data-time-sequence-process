import tensorflow as tf
import numpy as np
import json
import random

class OnlyAttention:
    def __init__(self, fixed_length, input_size, class_num, foresight_steps=0,
                 network_hyperparameters='./data/attention_network_hyperparameters.json'):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        self.fixed_length = fixed_length
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.input_size = input_size
        self.dtype = tf.float32
        self.class_num = class_num
        # self.parameters = []
        self.foresight_steps = foresight_steps
        with open(network_hyperparameters, 'r') as f:
            self.network_hyperparameter = json.load(f)

        # # todoo set parameter file format
        # attention_type = self.network_hyperparameter['todoo ']
        # self.attention_type = attention_type  # sdp for scaled dot-product attention, mh for multi-head attention

        return

    def _build_network(self):

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

    def _feature_attention(self, q, k, v, layer_name):
        # todo similar to position attention, do it before position embedding
        return

    def _multi_head_attention(self, q, k, v, layer_name):
        q = tf.reshape(q, [-1, q.shape[-1]], name='reshape_q')
        k = tf.reshape(k, [-1, k.shape[-1]], name='reshape_k')
        v = tf.reshape(v, [-1, v.shape[-1]], name='reshape_v')
        head_num = self.network_hyperparameter[layer_name]['head_num']
        head_size = self.network_hyperparameter[layer_name]['head_size']
        with tf.variable_scope(layer_name):
            linear_project_qw = tf.get_variable('linear_project_q', shape=[q.shape[-1], head_num*head_size],
                                               dtype=self.dtype,
                                               initializer=tf.contrib.layers.xavier_initializer())
            linear_project_kw = tf.get_variable('linear_project_k', shape=[k.shape[-1], head_num*head_size],
                                               dtype=self.dtype,
                                               initializer=tf.contrib.layers.xavier_initializer())
            linear_project_vw = tf.get_variable('linear_project_v', shape=[v.shape[-1], head_num * head_size],
                                               dtype=self.dtype,
                                               initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope('linear_project'):
                q = tf.matmul(q, linear_project_qw)
                k = tf.matmul(k, linear_project_kw)
                v = tf.matmul(v, linear_project_vw)

                q = tf.reshape(q, [-1, self.fixed_length, q.shape[-1]])
                k = tf.reshape(k, [-1, self.fixed_length, k.shape[-1]])
                v = tf.reshape(v, [-1, self.fixed_length, v.shape[-1]])

            with tf.name_scope('scaled_dot_product_attention'):
                new_vs = []
                for i in range(head_num):
                    temp_q = q[:, :, i*head_size: (i+1)*head_size]
                    temp_k = k[:, :, i*head_size: (i+1)*head_size]
                    attention = tf.matmul(temp_q, tf.transpose(temp_k, [0, 2, 1]))/tf.sqrt(head_size)
                    attention = tf.nn.softmax(attention, axis=2, name='attention%d' % i)
                    temp_v = v[:, :, i*head_size: (i+1)*head_size]
                    attention_v = tf.matmul(attention, temp_v)
                    new_vs.append(attention_v)

            with tf.name_scope('concat_linear_project'):
                concat = tf.concat(new_vs, axis=2)
                output_size = self.network_hyperparameter[layer_name]['output_size']  # d_model
                w = tf.get_variable('linear_project_concat', shape=[concat.shape[-1], output_size],
                                               dtype=self.dtype,
                                               initializer=tf.contrib.layers.xavier_initializer())
                output = tf.reshape(tf.matmul(tf.reshape(concat,
                                                         [-1, concat.shape[-1]]),
                                              w),
                                    [-1, self.fixed_length, output_size])

        return  output



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

        return 'one epoch done'

if __name__ == '__main__':
    oa =  OnlyAttention()