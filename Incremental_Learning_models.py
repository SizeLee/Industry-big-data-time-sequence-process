import tensorflow as tf
import numpy as np
import json
import random
import time


class Incremental_CNN_Attention:
    def __init__(self, fixed_length, input_size, class_num, foresight_steps=0,
                 network_hyperparameters='./data/attention_network_hyperparameters_v2.json',
                 incremental_net_hyperparameters='./data/Incremental_CNN_A.json'):
        self.graph = tf.Graph()

        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2

        self.sess = tf.Session(graph=self.graph, config=tf_config)

        self.sequence_length = fixed_length
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.input_size = input_size
        self.dtype = tf.float32
        self.class_num = class_num
        # self.parameters = []
        self.foresight_steps = foresight_steps
        with open(network_hyperparameters, 'r') as f:
            self.network_hyperparameter = json.load(f)

        self.incremental_net_hyperparameters = incremental_net_hyperparameters

        # # todoo set parameter file format
        # attention_type = self.network_hyperparameter['todoo ']
        # self.attention_type = attention_type  # sdp for scaled dot-product attention, mh for multi-head attention
        with self.graph.as_default():
            self._build_network()
            self._build_summary_node()
            self.initializer = tf.global_variables_initializer()
            self.sess.run(self.initializer)

        return

    def _build_network(self):
        with tf.variable_scope('input'):
            self.input = tf.placeholder(shape=[None, self.sequence_length, self.input_size],
                                        dtype=self.dtype, name='input_sequence')
            # shape = [sample, sequence_length, feature_num]

        # add incremental part
        layer_input, incre_var_list = self._incremental_map_layer(self.input,
                                                incremental_net_structure_file=self.incremental_net_hyperparameters)

        position_embedding = self._position_embedding(self.sequence_length, self.input_size, 'pre_process_layer')
        # position_embedding = self._position_embedding_v2(self.sequence_length, self.input_size, 'pre_process_layer')
        layer_input = layer_input + position_embedding
        self.check = position_embedding

        with tf.name_scope('cnn_layers'):
            layer_out = layer_input
            conv_layers_num = self.network_hyperparameter['cnn_layers_num']
            for i in range(conv_layers_num):
                layer_out = self._cnn_layer(i + 1, layer_out, self.network_hyperparameter)

        layer_input = layer_out

        with tf.name_scope('encoder_layers'):
            encoder_num = self.network_hyperparameter['encoder_num']
            for i in range(encoder_num):
                layer_out = self._attention_encoder_layer(layer_input, 'encoder_%d' % (i+1),
                                                    self.network_hyperparameter['encoders']['encoder_%d' % (i+1)])
                layer_input = layer_out

        linear_out = layer_out  # because summarizer attention layer return linear out

        with tf.name_scope('training_and_judging'):
            self.y = tf.placeholder(shape=[None, self.class_num], dtype=self.dtype, name='labels')
            self.weight_matrix = tf.placeholder(shape=[None, 1], dtype=self.dtype, name='weight_matrix')
            # print(tf.nn.softmax_cross_entropy_with_logits(logits=linear_out, labels=self.y).shape)
            # print((tf.nn.softmax_cross_entropy_with_logits(logits=linear_out, labels=self.y)
            #        * tf.reshape(self.weight_matrix, [-1])).shape)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=linear_out, labels=self.y)
                                      * tf.reshape(self.weight_matrix, [-1]))
            self.learning_rate = tf.placeholder(dtype=self.dtype, name='learning_rate')
            optimizer = tf.train.AdamOptimizer(self.learning_rate)  # learning rate could be adjust
            # set var_list, add incremental train step
            normal_part_var_list = list(set(tf.trainable_variables()) - set(incre_var_list))
            self.train_step = optimizer.minimize(self.loss, var_list=normal_part_var_list)
            incre_optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.incremental_train_step = incre_optimizer.minimize(self.loss, var_list=incre_var_list)

            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.predict, tf.argmax(self.y, axis=1)), dtype=self.dtype))

            self.batch_loss_summary = tf.summary.scalar('batch_loss', self.loss)
            self.batch_accuracy_summary = tf.summary.scalar('batch_accuracy', self.accuracy)
            self.batch_summary = tf.summary.merge([self.batch_loss_summary, self.batch_accuracy_summary])

        return

    def _incremental_map_layer(self, origin_input, incremental_net_structure_file='./data/Incremental_CNN_A.json'):
        with open(incremental_net_structure_file, 'r') as f:
            self.incremental_hyperparameter = json.load(f)

        with tf.variable_scope('incremental_layer') as scope:
            self.incremental_flag = tf.placeholder(tf.bool)
            self.incremental_input = tf.placeholder(shape=[None, self.sequence_length, self.input_size],
                                        dtype=self.dtype, name='input_sequence')

            with tf.variable_scope('cnn_layers'):
                layer_out = self.incremental_input
                conv_layers_num = self.network_hyperparameter['cnn_layers_num']
                for i in range(conv_layers_num):
                    layer_out = self._cnn_layer(i + 1, layer_out, self.incremental_hyperparameter)

            layer_out = tf.cond(self.incremental_flag, lambda: layer_out, lambda: origin_input)

        incremental_var_list = tf.trainable_variables()

        return layer_out, incremental_var_list

    def _build_summary_node(self):
        with tf.name_scope('summary_node'):
            self.whole_loss_node = tf.placeholder(tf.float32)
            self.whole_loss_summary = tf.summary.scalar('whole_loss', self.whole_loss_node)
            self.whole_accuracy_node = tf.placeholder(tf.float32)
            self.whole_accuracy_summary = tf.summary.scalar('whole_accuracy', self.whole_accuracy_node)
            self.whole_summary = tf.summary.merge([self.whole_loss_summary, self.whole_accuracy_summary])

    def train_v2(self, data, labels, samples_length, epoches, batch_size, train_set_sample_ids, test_set_ids,
                 learning_rate=0.001, foresight_steps=None, reset_flag=False, record_flag=True, incre_flag=False,
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
            train_writer = tf.summary.FileWriter(log_dir + '/sum_a/train', self.sess.graph)
            test_writer = tf.summary.FileWriter(log_dir + '/sum_a/test')


        no_sense_input = np.zeros((1, self.sequence_length, self.input_size))

        step_count = 0
        for i in range(epoches):
            # print('epoch%d:' % i)
            data_set = self._data_generator_v2(data, labels, self.sequence_length, samples_length, batch_size,
                                               train_set_sample_ids)
            for batch_data, batch_label, weight in data_set:
                # print(weight)
                if incre_flag:
                    loss, _, batch_summary = self.sess.run([self.loss, self.incremental_train_step, self.batch_summary],
                                                           feed_dict={self.input: no_sense_input, self.y: batch_label,
                                                                      self.learning_rate: learning_rate,
                                                                      self.weight_matrix: weight,
                                                                      self.incremental_flag: incre_flag,
                                                                      self.incremental_input: batch_data})
                    # print('step%d: %f' % (step_count, loss))
                else:
                    loss, _, batch_summary = self.sess.run([self.loss, self.train_step, self.batch_summary],
                                                        feed_dict={self.input: batch_data, self.y: batch_label,
                                                                   self.learning_rate: learning_rate,
                                                                   self.weight_matrix: weight,
                                                                   self.incremental_flag: incre_flag,
                                                                   self.incremental_input: no_sense_input})

                    # print('step%d: %f' % (step_count, loss))
                # print(check)

                if record_flag:
                    train_writer.add_summary(batch_summary, global_step=step_count)

                if step_count % 100 == 0 and record_flag:
                    self._whole_summary_write(train_writer, step_count, data, labels, samples_length, batch_size,
                                              train_set_sample_ids, incre_flag)

                    self._whole_summary_write(test_writer, step_count, data, labels, samples_length, batch_size,
                                              test_set_ids, incre_flag)

                step_count += 1

            accuracy, loss = self._cal_accuracy_and_loss_v2(data, labels, samples_length, batch_size, train_set_sample_ids, incre_flag)
            if loss < 0.1 and accuracy > 0.96:
                break
            # print()
            # self.sess.run(self.accuracy, feed_dict={})

        if record_flag:
            accuracy = self._whole_summary_write(train_writer, step_count, data, labels, samples_length, batch_size,
                                             train_set_sample_ids, incre_flag)

            self._whole_summary_write(test_writer, step_count, data, labels, samples_length, batch_size, test_set_ids, incre_flag)
        else:
            accuracy, _ = self._cal_accuracy_and_loss_v2(data, labels, samples_length, batch_size, train_set_sample_ids, incre_flag)

        # print('accuracy on training set: %f' % accuracy)
        # print(check)

        if record_flag:
            train_writer.close()
            test_writer.close()
        return accuracy

    def _whole_summary_write(self, writer, step, data, labels, samples_length, batch_size, data_set_sample_ids, incre_flag):
        w_accuracy, w_loss = self._cal_accuracy_and_loss_v2(data, labels, samples_length, batch_size,
                                                            data_set_sample_ids, incre_flag)
        whole_summary = self.sess.run(self.whole_summary, feed_dict={self.whole_accuracy_node: w_accuracy,
                                                                     self.whole_loss_node: w_loss})
        writer.add_summary(whole_summary, global_step=step)
        return w_accuracy

    def test_v2(self, data, label, samples_length, test_set_sample_ids=None, batch_size=1024, incre_flag=False, data_set_name='test set'):
        accuracy = self._cal_accuracy_v2(data, label, samples_length, batch_size, test_set_sample_ids, incre_flag)
        print('accuracy on %s: %f' % (data_set_name, accuracy))
        return accuracy

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

    def _attention_encoder_layer(self, layer_input, layer_name, net_structure):
        # layer_input[sample, fixed_length, features]
        with tf.name_scope(layer_name):
            attention_type = net_structure["attention_type"]
            if attention_type.lower() == 'mha':  # mha for multi-head attention, fa for feature attention
                mha = self._multi_head_attention(layer_input, layer_input, layer_input,
                                                 layer_name + '_multi-head_attention', net_structure['attention_layer'])
                if net_structure['attention_layer']['residual_flag'].lower() == 'true':
                    layer_out = mha + layer_input
                else:
                    layer_out = mha

            elif attention_type.lower() == 'fa':  # mha for multi-head attention, fa for feature attention
                fa = self._feature_attention(layer_input, layer_input, layer_input,
                                             'feature_attention', net_structure['attention_layer'])
                if net_structure['attention_layer']['residual_flag'].lower() == 'true':
                    layer_out = fa + layer_input
                else:
                    layer_out = fa

            elif attention_type.lower() == 'suma':
                if 'query_num' in net_structure:
                    query_num = net_structure['query_num']
                else:
                    query_num = 1
                suma = self._summarize_attention(layer_input, layer_input,
                                                 'summarizer_attention', net_structure['attention_layer'], query_num)
                if net_structure['attention_layer']['residual_flag'].lower() == 'true':
                    layer_out = suma + layer_input
                else:
                    layer_out = suma

            else:  # illegal attention type means no attention
                layer_out = layer_input

            # use position_wise or dense net, if position_wise in netstructure
            if 'position_wise_net' in net_structure:
                layer_out = self._position_wise_dense_layer(layer_out, '%s_position_wise_net' % layer_name,
                                                        net_structure['position_wise_net'])
                return layer_out

            elif 'cnn_layers' in net_structure:
                with tf.name_scope('cnn_layers'):
                    conv_layers_num = net_structure['cnn_layers_num']
                    for i in range(conv_layers_num):
                        layer_out = self._cnn_layer(i + 1, layer_out, net_structure)
                return layer_out

            else:
                if len(layer_out.shape) == 3:
                    layer_out = tf.reshape(layer_out, [-1, layer_out.shape[1] * layer_out.shape[2]])
                fc_layer_num = net_structure['fc_layer_num']
                linear_out = layer_out
                for i in range(fc_layer_num):
                    layer_out, linear_out = self._dense_layer('fc%d' % (i + 1), layer_out,
                                                            net_structure['fc_layers']['fc%d' % (i + 1)][
                                                                'size'],
                                                            net_structure['fc_layers']['fc%d' % (i + 1)][
                                                                'activation_func'])
                # out_put, linear_out = self._dense_layer('fc_out', out_put, self.class_num, 'softmax')
                self.predict = tf.argmax(layer_out, axis=1)

                return linear_out

    def _cnn_layer(self, layer_id, layer_input, net_structure):
        kernel_size = net_structure['cnn_layers']['layer_%d' % layer_id]['kernel_size']
        kernels_num = net_structure['cnn_layers']['layer_%d' % layer_id]['kernels_num']
        stride = net_structure['cnn_layers']['layer_%d' % layer_id]['stride']
        padding_pattern = net_structure['cnn_layers']['layer_%d' % layer_id]['padding']
        activation_pattern = net_structure['cnn_layers']['layer_%d' % layer_id]['activation'].lower()
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

    def _feature_attention(self, q, k, v, layer_name, net_structure):
        # similar to position attention, do it before position embedding
        head_num = net_structure['head_num']
        head_size = net_structure['head_size']
        sequence_length = q.shape[1]
        with tf.variable_scope(layer_name):
            q = tf.reshape(q, [-1, q.shape[-1]], name='reshape_q')
            k = tf.reshape(k, [-1, k.shape[-1]], name='reshape_k')
            v = tf.reshape(v, [-1, v.shape[-1]], name='reshape_v')
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

                q = tf.reshape(q, [-1, sequence_length, q.shape[-1]])
                k = tf.reshape(k, [-1, sequence_length, k.shape[-1]])
                v = tf.reshape(v, [-1, sequence_length, v.shape[-1]])

            with tf.name_scope('scaled_dot_product_attention'):
                q = tf.concat(tf.split(q, head_num, axis=2), axis=0)
                k = tf.concat(tf.split(k, head_num, axis=2), axis=0)
                v = tf.concat(tf.split(v, head_num, axis=2), axis=0)  # [sample*head_num, fixed_length, head_size]
                attention = tf.matmul(tf.transpose(q, [0, 2, 1]), k) / (head_size ** 0.5)
                attention = tf.nn.softmax(attention, axis=1, name='attention')
                attention_v = tf.matmul(v, attention)
                # new_vs = []
                # for i in range(head_num):
                #     temp_q = q[:, :, i*head_size: (i+1)*head_size]
                #     temp_k = k[:, :, i*head_size: (i+1)*head_size]
                #     attention = tf.matmul(tf.transpose(temp_q, [0, 2, 1]), temp_k)/tf.sqrt(float(head_size))
                #     attention = tf.nn.softmax(attention, axis=1, name='attention%d' % i)
                #     temp_v = v[:, :, i*head_size: (i+1)*head_size]
                #     attention_v = tf.matmul(temp_v, attention)
                #     new_vs.append(attention_v)

            with tf.name_scope('concat_linear_project'):
                # concat = tf.concat(new_vs, axis=2)
                concat = tf.concat(tf.split(attention_v, head_num, axis=0), axis=2)  # [sample, fixed_length, head_num*head_size]
                output_size = net_structure['output_size']  # d_model
                w = tf.get_variable('linear_project_concat', shape=[concat.shape[-1], output_size],
                                               dtype=self.dtype,
                                               initializer=tf.contrib.layers.xavier_initializer())
                output = tf.reshape(tf.matmul(tf.reshape(concat,
                                                         [-1, concat.shape[-1]]),
                                              w),
                                    [-1, sequence_length, output_size])

        return output

    def _multi_head_attention(self, q, k, v, layer_name, net_structure):
        head_num = net_structure['head_num']
        head_size = net_structure['head_size']
        sequence_length = q.shape[1]
        with tf.variable_scope(layer_name):
            q = tf.reshape(q, [-1, q.shape[-1]], name='reshape_q')
            k = tf.reshape(k, [-1, k.shape[-1]], name='reshape_k')
            v = tf.reshape(v, [-1, v.shape[-1]], name='reshape_v')
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

                q = tf.reshape(q, [-1, sequence_length, q.shape[-1]])
                k = tf.reshape(k, [-1, sequence_length, k.shape[-1]])
                v = tf.reshape(v, [-1, sequence_length, v.shape[-1]])

            with tf.name_scope('scaled_dot_product_attention'):
                q = tf.concat(tf.split(q, head_num, axis=2), axis=0)
                k = tf.concat(tf.split(k, head_num, axis=2), axis=0)
                v = tf.concat(tf.split(v, head_num, axis=2), axis=0)  # [sample*head_num, fixed_length, head_size]
                attention = tf.matmul(q, tf.transpose(k, [0, 2, 1]))/(head_size ** 0.5)
                attention = tf.nn.softmax(attention, axis=2, name='attention')
                attention_v = tf.matmul(attention, v)
                # new_vs = []
                # for i in range(head_num):
                #     temp_q = q[:, :, i*head_size: (i+1)*head_size]
                #     temp_k = k[:, :, i*head_size: (i+1)*head_size]
                #     attention = tf.matmul(temp_q, tf.transpose(temp_k, [0, 2, 1]))/tf.sqrt(float(head_size))
                #     attention = tf.nn.softmax(attention, axis=2, name='attention%d' % i)
                #     temp_v = v[:, :, i*head_size: (i+1)*head_size]
                #     attention_v = tf.matmul(attention, temp_v)
                #     new_vs.append(attention_v)

            with tf.name_scope('concat_linear_project'):
                # concat = tf.concat(new_vs, axis=2)
                concat = tf.concat(tf.split(attention_v, head_num, axis=0), axis=2)  # [sample, fixed_length, head_num*head_size]
                output_size = net_structure['output_size']  # d_model
                w = tf.get_variable('linear_project_concat', shape=[concat.shape[-1], output_size],
                                               dtype=self.dtype,
                                               initializer=tf.contrib.layers.xavier_initializer())
                output = tf.reshape(tf.matmul(tf.reshape(concat,
                                                         [-1, concat.shape[-1]]),
                                              w),
                                    [-1, sequence_length, output_size])

        return output  # output shape [sample, fixed_length, output_size]

    def _summarize_attention(self, k, v, layer_name, net_structure, query_num=1):
        # todo add more query vectors
        head_num = net_structure['head_num']
        head_size = net_structure['head_size']
        sequence_length = k.shape[1]
        sample_num = tf.shape(k)[0]
        with tf.variable_scope(layer_name):
            q = tf.get_variable('q', shape=[query_num, k.shape[-1]], dtype=self.dtype,
                                initializer=tf.random_normal_initializer(stddev=0.5))
            k = tf.reshape(k, [-1, k.shape[-1]], name='reshape_k')
            v = tf.reshape(v, [-1, v.shape[-1]], name='reshape_v')

            linear_project_qw = tf.get_variable('linear_project_q', shape=[q.shape[-1], head_num * head_size],
                                                dtype=self.dtype,
                                                initializer=tf.contrib.layers.xavier_initializer())
            linear_project_kw = tf.get_variable('linear_project_k', shape=[k.shape[-1], head_num * head_size],
                                                dtype=self.dtype,
                                                initializer=tf.contrib.layers.xavier_initializer())
            linear_project_vw = tf.get_variable('linear_project_v', shape=[v.shape[-1], head_num * head_size],
                                                dtype=self.dtype,
                                                initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope('linear_project'):
                q = tf.matmul(q, linear_project_qw)
                k = tf.matmul(k, linear_project_kw)
                v = tf.matmul(v, linear_project_vw)
                q = tf.tile(tf.expand_dims(q, axis=0), [sample_num, 1, 1])  # [sample, query_num, head_size*head_num]
                k = tf.reshape(k, [-1, sequence_length, k.shape[-1]])
                v = tf.reshape(v, [-1, sequence_length, v.shape[-1]])

            with tf.name_scope('summarizer_attention'):
                q = tf.concat(tf.split(q, head_num, axis=2), axis=0)  # [sample*head_num, query_num, head_size]
                k = tf.concat(tf.split(k, head_num, axis=2), axis=0)  # [sample*head_num, fixed_length, head_size]
                v = tf.concat(tf.split(v, head_num, axis=2), axis=0)  # [sample*head_num, fixed_length, head_size]
                attention = tf.matmul(q, tf.transpose(k, [0, 2, 1])) / (head_size ** 0.5)
                attention = tf.nn.softmax(attention, axis=2, name='attention')
                attention_v = tf.matmul(attention, v)  # [sample*head_num, query_num, head_size]
                # new_vs = []
                # for i in range(head_num):
                #     temp_q = q[:, :, i*head_size: (i+1)*head_size]
                #     temp_k = k[:, :, i*head_size: (i+1)*head_size]
                #     attention = tf.matmul(temp_q, tf.transpose(temp_k, [0, 2, 1]))/tf.sqrt(float(head_size))
                #     attention = tf.nn.softmax(attention, axis=2, name='attention%d' % i)
                #     temp_v = v[:, :, i*head_size: (i+1)*head_size]
                #     attention_v = tf.matmul(attention, temp_v)
                #     new_vs.append(attention_v)

            with tf.name_scope('concat_linear_project'):
                # concat = tf.concat(new_vs, axis=2)
                concat = tf.concat(tf.split(attention_v, head_num, axis=0), axis=2)  # [sample, query_num, head_num*head_size]
                output_size = net_structure['output_size']  # d_model
                w = tf.get_variable('linear_project_concat', shape=[concat.shape[-1], output_size],
                                               dtype=self.dtype,
                                               initializer=tf.contrib.layers.xavier_initializer())
                output = tf.reshape(tf.matmul(tf.reshape(concat,
                                                         [-1, concat.shape[-1]]),
                                              w),
                                    [-1, query_num, output_size])

        return output  # output shape [sample, query_num, output_size]

    def _position_wise_dense_layer(self, layer_input, layer_name, network_structure):
        w1_size = network_structure['w1_size']
        w2_size = network_structure['w2_size']
        sequence_length = layer_input.shape[1]
        with tf.variable_scope(layer_name):
            w1 = tf.get_variable('weight1', shape=[layer_input.shape[-1], w1_size],
                                               dtype=self.dtype,
                                               initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(initial_value=tf.constant(0, shape=[w1_size], dtype=self.dtype), name='bias1')
            output = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.reshape(layer_input, [-1, layer_input.shape[-1]]), w1), b))

            w2 = tf.get_variable('weight2', shape=[w1_size, w2_size],
                                               dtype=self.dtype,
                                               initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(initial_value=tf.constant(0, shape=[w2_size], dtype=self.dtype), name='bias2')
            output = tf.nn.relu(tf.nn.bias_add(tf.matmul(output, w2), b))

            output = tf.reshape(output, [-1, sequence_length, w2_size])

            if network_structure['residual_flag'].lower() == 'true':
                output = output + layer_input

        return output

    def _dense_layer(self, layer_name, layer_input, layer_out_size, activation_func='relu',
                     reuse_flag=False, reuse_w=None, reuse_b=None):
        with tf.name_scope(layer_name):
            with tf.variable_scope(layer_name+'_variables'):
                # w = tf.Variable(initial_value=tf.truncated_normal([layer_input.shape[1], layer_out_size], stddev=1e-1,
                #                                                   dtype=self.dtype), name='weight')
                if reuse_flag and reuse_w is not None and reuse_b is not None:
                    w = reuse_w
                    b = reuse_b
                else:
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

    def _position_embedding(self, length, channels, layer_name, max_time_scale=1.0e4, min_time_scale=1.0, start_index=0):
        with tf.name_scope('%s_position_embedding' % layer_name):
            position = tf.cast(tf.range(length) + start_index, dtype=self.dtype)
            dimention = tf.cast(tf.range(channels // 2), dtype=self.dtype)

            signal = tf.expand_dims(position, 1) / tf.pow(float(max_time_scale) / float(min_time_scale),
                                                          2 * tf.expand_dims(dimention, 0) / float(channels))

            sin_signal = tf.sin(signal)
            cos_signal = tf.cos(signal)

            signal = tf.reshape(tf.concat([tf.reshape(sin_signal, [-1, 1]), tf.reshape(cos_signal, [-1, 1])], axis=1),
                                [-1, (channels // 2) * 2])

            if channels % 2 == 1:
                last_embedding = tf.expand_dims(position, 1) \
                                 / tf.pow(float(max_time_scale) / float(min_time_scale),
                                       2 * tf.constant(channels // 2, shape=[1, 1], dtype=self.dtype) / float(channels))
                signal = tf.concat([signal, tf.sin(last_embedding)], axis=-1)

            signal = tf.reshape(signal, [1, length, channels])
        return signal

    def _position_embedding_v2(self, length, channels, layer_name, max_time_scale=1.0e4, min_time_scale=1.0, start_index=0):
        with tf.name_scope('%s_position_embedding' % layer_name):
            position = tf.cast(tf.range(length) + start_index, dtype=self.dtype)
            dimention = tf.cast(tf.range(channels // 2), dtype=self.dtype)

            signal = tf.expand_dims(position, 1) / tf.pow(float(max_time_scale) / float(min_time_scale),
                                                          2 * tf.expand_dims(dimention, 0) / float(channels))

            sin_signal = tf.sin(signal)
            cos_signal = tf.cos(signal)

            signal = tf.reshape(tf.concat([tf.reshape(sin_signal, [-1, 1]), tf.reshape(cos_signal, [-1, 1])], axis=1),
                                [-1, (channels // 2) * 2])

            if channels % 2 == 1:
                last_embedding = tf.expand_dims(position, 1) \
                                 / tf.pow(float(max_time_scale) / float(min_time_scale),
                                       2 * tf.constant(channels // 2, shape=[1, 1], dtype=self.dtype) / float(channels))
                signal = tf.concat([signal, tf.sin(last_embedding)], axis=-1)

            signal = tf.reshape(signal, [1, length, channels])

            signal = tf.get_variable('changeable_position_embedding', initializer=signal)

        return signal

    def _cal_accuracy_v2(self, data, labels, samples_length, batch_size, sample_ids=None, incre_flag=False):
        data_set = self._data_generator_v2(data, labels, self.sequence_length, samples_length, batch_size, sample_ids)
        batch_count = 0
        accuracy = 0.
        no_sense_input = np.zeros((1, self.sequence_length, self.input_size))
        for batch_data, batch_label, _ in data_set:
            batch_count += batch_label.shape[0]
            if incre_flag:
                accuracy += batch_label.shape[0] * self.sess.run(self.accuracy,
                                                                 feed_dict={self.input: no_sense_input,
                                                                            self.y: batch_label,
                                                                            self.incremental_input: batch_data,
                                                                            self.incremental_flag: incre_flag})
            else:
                accuracy += batch_label.shape[0] * self.sess.run(self.accuracy,
                                                             feed_dict={self.input: batch_data, self.y: batch_label,
                                                                        self.incremental_input: no_sense_input,
                                                                        self.incremental_flag: incre_flag})
        accuracy /= batch_count
        return accuracy

    def _cal_accuracy_and_loss_v2(self, data, labels, samples_length, batch_size, sample_ids=None, incre_flag=False):
        data_set = self._data_generator_v2(data, labels, self.sequence_length, samples_length, batch_size, sample_ids)
        sample_count = 0
        accuracy = 0.
        loss = 0.
        no_sense_input = np.zeros((1, self.sequence_length, self.input_size))
        for batch_data, batch_label, weight in data_set:
            sample_count += batch_label.shape[0]
            if incre_flag:
                b_ac, b_loss = self.sess.run([self.accuracy, self.loss],
                                             feed_dict={self.input: no_sense_input, self.y: batch_label,
                                                        self.weight_matrix: weight, self.incremental_input: batch_data,
                                                        self.incremental_flag: incre_flag})
            else:
                b_ac, b_loss = self.sess.run([self.accuracy, self.loss],
                                         feed_dict={self.input: batch_data, self.y: batch_label,
                                                    self.weight_matrix: weight, self.incremental_input: no_sense_input,
                                                    self.incremental_flag: incre_flag})
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

    # def test_time(self, data, labels, sample_length, test_set_ids, save_dir='./data/confusion_matrix/cnn'):
    #     whole_time, y_true, y_predict = self._get_test_result_and_running_time(data, labels, sample_length, 1024,
    #                                                                            test_set_ids)
    #     np.savez(save_dir + '/sum_a_cnn_predict.npz', y_predict=y_predict, y_true=y_true)
    #     return whole_time
    #
    # def _get_test_result_and_running_time(self, data, labels, samples_length, batch_size, sample_ids=None):
    #     data_set = self._data_generator_v2(data, labels, self.sequence_length, samples_length, batch_size, sample_ids)
    #     y_true = []
    #     y_predict = []
    #     whole_time = 0.
    #     for batch_data, batch_label, _ in data_set:
    #         b_true = np.argmax(batch_label, axis=1)
    #         start = time.time()
    #         b_pre = self.sess.run(self.predict, feed_dict={self.input: batch_data, self.y: batch_label})
    #         b_pre = b_pre.reshape([-1])
    #         end = time.time()
    #         whole_time += end - start
    #         y_true.append(b_true)
    #         y_predict.append(b_pre)
    #
    #     y_true = np.concatenate(y_true)
    #     y_predict = np.concatenate(y_predict)
    #
    #     return whole_time, y_true, y_predict

    def incremental_simulation(self, data, labels, samples_length, epoches, batch_size, train_set_sample_ids,
                               test_set_ids, incremental_set_ids, learning_rate=0.001, foresight_steps=None,
                               reset_flag=False, record_flag=True, log_dir='./data/log/i_cnn_models', random_seed=None):

        # self.train_v2(data, labels, samples_length, epoches, batch_size, train_set_sample_ids, test_set_ids,
        #               learning_rate=learning_rate, foresight_steps=foresight_steps, reset_flag=reset_flag,
        #               record_flag=record_flag, log_dir=log_dir, random_seed=random_seed)
        #
        # self.save_model('./model/incre_cnn_a')

        self.load_model('./model/incre_cnn_a')

        self.test_v2(data, labels, samples_length, test_set_sample_ids=test_set_ids, batch_size=1024, data_set_name='test set')

        #### origin train over
        bin_upper = np.max(data, axis=0)
        bin_lower = np.min(data, axis=0)

        bin_num = 20

        bins_bound = np.zeros((data.shape[-1], bin_num + 1))

        for i in range(data.shape[-1]):
            bins_bound[i, :] = np.linspace(bin_lower[i] - 1e-3, bin_upper[i] + 1e-3, bin_num + 1)

        # print(bins_bound)

        origin_data = data[train_set_sample_ids + test_set_ids, :]
        origin_distribution = self._bining_gen_distribution(origin_data, bins_bound)

        new_data = data[incremental_set_ids, :]
        new_distribution = self._bining_gen_distribution(new_data, bins_bound)
        f_d = self._distribution_similarity_judge(origin_distribution, new_distribution)
        # print(f_d)
        print('KL difference between old set and new set')
        print('mean:', f_d.mean())
        print('max:', f_d.max())
        print('min', f_d.min())
        print('median', np.median(f_d))
        # f_d = self._distribution_similarity_judge(origin_distribution, origin_distribution)
        # print(f_d)
        print(' ')

        distribution_difference_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        metricses = {'mean': np.mean, 'max': np.max, 'median': np.median, 'min': np.min}
        # metricses = {'median': np.median}
        # distribution_difference_threshold = 0.6
        lines = {}
        for metrics_name in metricses:
            metrics = metricses[metrics_name]
            lines[metrics_name] = {}
            for distribution_difference_threshold in distribution_difference_thresholds:
                lines[metrics_name][distribution_difference_threshold] = {'ac': [], 'up': []}
                self.sess.run(self.initializer)
                self.load_model('./model/incre_cnn_a')
                old_distribution = origin_distribution
                print(' ')
                print('exam of %s, %f:' % (metrics_name, distribution_difference_threshold))
                slide_window = 10000
                slide_step = 10000
                window_start = 0
                incre_flag = False
                dis_from_update = 0
                while window_start < len(incremental_set_ids):
                    new_data_ids = incremental_set_ids[window_start: window_start + slide_window]
                    new_data = data[new_data_ids, :]
                    new_distribution = self._bining_gen_distribution(new_data, bins_bound)
                    f_d = self._distribution_similarity_judge(old_distribution, new_distribution)
                    # print(f_d.mean(), f_d.max())
                    if metrics(f_d) > distribution_difference_threshold:
                        incre_flag = True
                        ac = self.train_v2(data, labels, samples_length, 23, batch_size, new_data_ids, None,
                                      learning_rate=learning_rate, foresight_steps=foresight_steps, reset_flag=False,
                                      record_flag=False, incre_flag=incre_flag, log_dir=log_dir,
                                      random_seed=random_seed)
                        # test_set_ids can be None when record_flag = False
                        print('1w start from %d sample' % window_start +' training accuracy: %f' % ac)

                        old_distribution = new_distribution
                        dis_from_update = 0
                        lines[metrics_name][distribution_difference_threshold]['ac'].append(ac)
                        lines[metrics_name][distribution_difference_threshold]['up'].append(dis_from_update)
                    else:
                        dis_from_update += 1
                        ac = self.test_v2(data, labels, samples_length, new_data_ids, incre_flag=incre_flag,
                                     data_set_name='1w start from %d sample' % window_start)
                        lines[metrics_name][distribution_difference_threshold]['ac'].append(ac)
                        lines[metrics_name][distribution_difference_threshold]['up'].append(dis_from_update)
                    window_start += slide_step
                    print()
                print('end exam of %s, %f:' % (metrics_name, distribution_difference_threshold))
                print(' ')
            print('end of %s' % metrics_name)

        with open('./data/incremental_result.json', 'w') as f:
            json.dump(lines, f)
        # with open('./data/incremental_median_result.json', 'w') as f:
        #     json.dump(lines, f)

        # slide_window = 10000
        # slide_step = 10000
        # window_start = 0
        # incre_flag = False
        # while window_start < len(incremental_set_ids):
        #     new_data_ids = incremental_set_ids[window_start: window_start+slide_window]
        #     new_data = data[new_data_ids, :]
        #     new_distribution = self._bining_gen_distribution(new_data, bins_bound)
        #     f_d = self._distribution_similarity_judge(origin_distribution, new_distribution)
        #     print(f_d.mean(), f_d.max())
        #     if f_d.mean() > distribution_difference_threshold:
        #         incre_flag = True
        #         self.train_v2(data, labels, samples_length, 23, batch_size, new_data_ids,  None,
        #               learning_rate=learning_rate, foresight_steps=foresight_steps, reset_flag=False,
        #               record_flag=False, incre_flag=incre_flag, log_dir=log_dir, random_seed=random_seed)
        #         # test_set_ids can be None when record_flag = False
        #         origin_distribution = new_distribution
        #     else:
        #         self.test_v2(data, labels, samples_length, new_data_ids, incre_flag=incre_flag,
        #                      data_set_name='1w start from %d sample' % window_start)
        #     window_start += slide_step
        #     print()

        return

    def _bining_gen_distribution(self, data, bins):
        feature_num = data.shape[-1]
        bin_num = bins.shape[-1] - 1
        distribution = np.ones((feature_num, bin_num)) * 0.1
        data = np.sort(data, axis=0)

        for fi in range(feature_num):
            count = 0
            cur_bin = 0
            si = 0
            while si < data.shape[0]:
                if data[si, fi] >= bins[fi, cur_bin + 1]:
                    distribution[fi, cur_bin] += count
                    count = 0
                    cur_bin += 1
                else:
                    count += 1
                    si += 1

            if count != 0:
                distribution[fi, cur_bin] += count

        distribution = distribution/np.sum(distribution, axis=-1, keepdims=True)  # shape is [feature_num, bin_num]
        # print(distribution)

        return distribution

    def _f(self, t, measure_way='KL'):
        if measure_way.upper() == 'KL':
            return t * np.log(t)
        elif measure_way.lower() == 'chi':
            return (t - 1) ** 2
        elif measure_way.lower() == 'reversekl':
            return -np.log(t)

    def _distribution_similarity_judge(self, data_set1_distribution, data_set2_distribution, measure_way='KL'):
        f_divergence = np.sum(data_set2_distribution * self._f(data_set1_distribution / data_set2_distribution, measure_way), axis=-1)
        return f_divergence


