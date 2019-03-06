import numpy as np
import configparser
import json
import heapq
import tensorflow as tf

class KNN_Sequence:
    def __init__(self):
        self.graph = None
        self.sess = None
        self.dtype = tf.float32
        return

    def train(self, training_data, labels, train_set_sample_ids, samples_length):
        self.training_data = training_data
        self.labels = labels
        self.train_set_sample_ids = train_set_sample_ids
        self.samples_length = samples_length

        data_id_by_length = {}

        for i in train_set_sample_ids:
            if self.samples_length[i, 0] not in data_id_by_length:
                data_id_by_length[self.samples_length[i, 0]] = []
            data_id_by_length[self.samples_length[i, 0]].append(i)

        for each in data_id_by_length:
            sample_num = len(data_id_by_length[each])
            sample_len = int(each)
            print(str(each)+':', sample_num)
            part_features = np.zeros((sample_num, sample_len, self.training_data.shape[1]))
            part_labels = np.zeros((sample_num, 1))
            for i, sample in enumerate(data_id_by_length[each]):
                part_features[i, :, :] = self.training_data[sample:sample+sample_len, :]
                part_labels[i, 0] = self.labels[sample, 0]

            file_name = './data/knn/l_%d.npz' % sample_len
            np.savez(file_name, features=part_features, labels=part_labels)

        return

    def test_cpu(self, data, labels, test_set_ids, samples_length, k):
        data_id_by_length = {}

        for i in test_set_ids:
            if samples_length[i, 0] not in data_id_by_length:
                data_id_by_length[samples_length[i, 0]] = []
            data_id_by_length[samples_length[i, 0]].append(i)


        final_labels = []
        final_predicts = []
        all_part_accuracy = []

        for each_length in data_id_by_length:
            ids = data_id_by_length[each_length]
            sample_num = len(ids)
            sample_len = int(each_length)
            print(str(each_length) + ':', sample_num)
            part_features = np.zeros((sample_num, sample_len, data.shape[1]))
            part_labels = np.zeros((sample_num, 1))
            for i, sample in enumerate(ids):
                part_features[i, :, :] = data[sample:sample + sample_len, :]
                part_labels[i, 0] = labels[sample, 0]
            predict_labels = self.predict_cpu(part_features, sample_len, k)

            final_labels.append(part_labels)
            final_predicts.append(predict_labels)
            part_accuracy = np.mean((part_labels == predict_labels)*1.)
            print('L%d accuracy: %f' % (sample_len, part_accuracy))
            all_part_accuracy.append(part_accuracy)

        whole_accuracy = 0.
        for i in range(len(all_part_accuracy)):
            whole_accuracy += all_part_accuracy[i] * final_labels[i].shape[0]
        whole_accuracy /= len(test_set_ids)
        print('whole_accuracy: %f' % whole_accuracy)
        self.sess.close()
        return

    def predict_cpu(self, data_features, sample_len, k):
        sample_len = int(sample_len)
        saved_data = np.load('./data/knn/l_%d.npz' % sample_len)
        saved_features = saved_data['features']
        saved_labels = saved_data['labels']
        test_sample_num = data_features.shape[0]
        print('samples num to compare %d of L%d' % (saved_labels.shape[0], sample_len))
        test_labels = np.zeros((test_sample_num, 1))
        for i in range(test_sample_num):
            print('\x1B[1A\x1B[K%d/%d\r' % (i, test_sample_num))
            for_test = data_features[i, :, :]
            distance = np.sum(np.sum((for_test-saved_features) ** 2, axis=2), axis=1)
            print(distance.shape)
            nearest_index = heapq.nsmallest(k, range(saved_features.shape[0]), distance.take)
            nearest_labels = saved_labels[nearest_index, :]

            count = {}
            for j in range(k):
                count[nearest_labels[j, 0]] = count.get(nearest_labels[j, 0], 0) + 1
            max_label = None
            max_time = 0
            for each in count:
                if count[each] > max_time:
                    max_label = each
            test_labels[i, 0] = max_label
        print('test L%d over.' % sample_len)

        return test_labels


    def test(self, data, labels, test_set_ids, samples_length, k):
        data_id_by_length = {}
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        for i in test_set_ids:
            if samples_length[i, 0] not in data_id_by_length:
                data_id_by_length[samples_length[i, 0]] = []
            data_id_by_length[samples_length[i, 0]].append(i)

        self.tf_cal_node = {}
        with self.graph.as_default():
            for each_length in data_id_by_length:
                sample_len = int(each_length)
                self.tf_cal_node[sample_len] = self._build_distance_calculate_top_k(sample_len, data.shape[1], k)

                # saved_data = np.load('./data/knn/l_%d.npz' % sample_len)
                # saved_features = saved_data['features']
                # self.tf_cal_node[sample_len] = self._build_distance_calculate_top_k_v2(sample_len, data.shape[1], k,
                #                                                                 saved_features)
            # self.sess.run(tf.global_variables_initializer())

        print(self.tf_cal_node.keys())

        final_labels = []
        final_predicts = []
        all_part_accuracy = []

        for each_length in data_id_by_length:
            ids = data_id_by_length[each_length]
            sample_num = len(ids)
            sample_len = int(each_length)
            print(str(each_length) + ':', sample_num)
            part_features = np.zeros((sample_num, sample_len, data.shape[1]))
            part_labels = np.zeros((sample_num, 1))
            for i, sample in enumerate(ids):
                part_features[i, :, :] = data[sample:sample + sample_len, :]
                part_labels[i, 0] = labels[sample, 0]
            predict_labels = self.predict(part_features, sample_len, k)

            final_labels.append(part_labels)
            final_predicts.append(predict_labels)
            part_accuracy = np.mean((part_labels == predict_labels)*1.)
            print('L%d accuracy: %f' % (sample_len, part_accuracy))
            all_part_accuracy.append(part_accuracy)

        whole_accuracy = 0.
        for i in range(len(all_part_accuracy)):
            whole_accuracy += all_part_accuracy[i] * final_labels[i].shape[0]
        whole_accuracy /= len(test_set_ids)
        print('whole_accuracy: %f' % whole_accuracy)
        self.sess.close()
        return

    def predict(self, data_features, sample_len, k):
        sample_len = int(sample_len)
        saved_data = np.load('./data/knn/l_%d.npz' % sample_len)
        saved_features = saved_data['features']
        saved_labels = saved_data['labels']
        test_sample_num = data_features.shape[0]
        print('samples num to compare %d of L%d' % (saved_labels.shape[0], sample_len))
        cal_node = self.tf_cal_node[sample_len]
        test_labels = np.zeros((test_sample_num, 1))
        for i in range(test_sample_num):
            # print('\x1B[1A\x1B[K%d/%d\r' % (i, test_sample_num))
            print('%d/%d' % (i, test_sample_num))
            for_test = data_features[i, :, :]
            # distance = np.sum(np.sum((for_test-saved_features) ** 2, axis=2), axis=1)
            # print(distance.shape)
            # nearest_index = heapq.nsmallest(k, range(saved_features.shape[0]), distance.take)

            nearest_index = self.sess.run(cal_node['topk_indices'], feed_dict={cal_node['for_test']: for_test,
                                                                               cal_node['saved_features']: saved_features})
            # nearest_index = self.sess.run(cal_node['topk_indices'], feed_dict={cal_node['for_test']: for_test})

            nearest_labels = saved_labels[nearest_index, :]

            count = {}
            for j in range(k):
                count[nearest_labels[j, 0]] = count.get(nearest_labels[j, 0], 0) + 1
            max_label = None
            max_time = 0
            for each in count:
                if count[each] > max_time:
                    max_label = each
            test_labels[i, 0] = max_label
        print('test L%d over.' % sample_len)

        return test_labels

    def _build_distance_calculate_top_k_v2(self, sample_length, features_size, k, saved_features):
        for_test = tf.placeholder(self.dtype, shape=[sample_length, features_size], name='L%d_place_holder' % sample_length)
        saved_features = tf.Variable(initial_value=saved_features, dtype=self.dtype, name='L%d_saved_features' % sample_length)
        distance = tf.reduce_sum(tf.pow(saved_features-for_test, 2), axis=[2, 1])
        top_nearest = tf.nn.top_k(-distance, k)
        index = top_nearest.indices
        return {'for_test': for_test, 'topk_indices': index}

    def _build_distance_calculate_top_k(self, sample_length, features_size, k):
        for_test = tf.placeholder(self.dtype, shape=[sample_length, features_size])
        saved_features = tf.placeholder(self.dtype, shape=[None, sample_length, features_size])
        distance = tf.reduce_sum(tf.pow(saved_features-for_test, 2), axis=[2, 1])
        top_nearest = tf.nn.top_k(-distance, k)
        index = top_nearest.indices
        return {'for_test': for_test, 'saved_features': saved_features, 'topk_indices': index}



if __name__ == '__main__':
    common_para = configparser.ConfigParser()
    common_para.read('common_para.ini')
    sequence_fix_length = common_para['common_parameters'].getint('sequence_fix_length')
    foresight_steps = common_para['common_parameters'].getint('foresight_steps')
    class_num = common_para['common_parameters'].getint('class_num')

    data = np.load(common_para['path']['data_file'])
    # print(data['features'])
    # print(data['labels'])
    # print(data['samples_length'])

    ### generate training samples' id

    with open(common_para['path']['data_set_ids_file'], 'r') as f:
        data_set_ids = json.load(f)
    training_sample_ids = data_set_ids['training_set']
    test_sample_ids = data_set_ids['test_set']

    knn = KNN_Sequence()
    # knn.train(data['features'], data['labels'], training_sample_ids, data['samples_length'])
    knn.test_cpu(data['features'], data['labels'], test_sample_ids[:100000], data['samples_length'], 100)
