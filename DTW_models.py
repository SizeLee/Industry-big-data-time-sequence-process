import numpy as np
import configparser
import json
import heapq
import random
import time


class DTW:
    def __init__(self):

        return

    def train(self, training_data, labels, train_set_sample_ids, samples_length, sampling_ratio=0.001):
        self.training_data = training_data
        self.labels = labels
        self.train_set_sample_ids = train_set_sample_ids
        self.samples_length = samples_length

        self.record_sample_ids = random.sample(training_sample_ids, int(len(training_sample_ids)*sampling_ratio))
        # print(len(record_sample_ids))

        data_id_by_length = {}

        for i in self.record_sample_ids:
            if self.samples_length[i, 0] not in data_id_by_length:
                data_id_by_length[self.samples_length[i, 0]] = []
            data_id_by_length[self.samples_length[i, 0]].append(i)

        for each in data_id_by_length:
            sample_num = len(data_id_by_length[each])
            sample_len = int(each)
            print(str(each) + ':', sample_num)
            part_features = np.zeros((sample_num, sample_len, self.training_data.shape[1]))
            part_labels = np.zeros((sample_num, 1))
            for i, sample in enumerate(data_id_by_length[each]):
                part_features[i, :, :] = self.training_data[sample:sample + sample_len, :]
                part_labels[i, 0] = self.labels[sample, 0]

            file_name = './data/dtw/l_%d.npz' % sample_len
            np.savez(file_name, features=part_features, labels=part_labels)

        return

    def predict(self, data_features, records_dict, k=1, window_size=15):
        # predict by all the records with different lengths on one sample
        predict_labels = []
        predict_distance = []
        for sample_len in range(16, 31):
            saved_features = records_dict[sample_len][0]
            saved_labels = records_dict[sample_len][1]
            nearest_index, nearest_distance = self._cal_nearest_by_dtw(data_features, saved_features, k, window_size,
                                                                       cal_version=1)
            nearest_labels = saved_labels[nearest_index, :].reshape((-1)).tolist()
            nearest_distance = nearest_distance.reshape((-1)).tolist()
            predict_labels.extend(nearest_labels)
            predict_distance.extend(nearest_distance)

        # print(predict_distance)
        # print(predict_labels)
        # print()
        nearest_index = heapq.nsmallest(k, range(len(predict_distance)), key=lambda x: predict_distance[x])
        count = {}
        for j in range(k):
            count[predict_labels[nearest_index[j]]] = count.get(predict_labels[nearest_index[j]], 0) + 1
        max_label = None
        max_time = 0
        for each in count:
            if count[each] > max_time:
                max_label = each
                max_time = count[each]

        return max_label

    def test(self, data, labels, test_set_ids, samples_length, k=1, window_size=15):
        # generate one data, judge once
        # save all records in memory, 10 percent occupies less than 1.44GB
        # count accuracy when judge

        records_dict = {}
        for sample_len in range(16, 31):
            saved_data = np.load('./data/dtw/l_%d.npz' % sample_len)
            saved_features = saved_data['features']
            saved_labels = saved_data['labels']
            records_dict[sample_len] = (saved_features, saved_labels)

        right_count = 0
        count = 0

        start = time.time()

        pre_labels = []
        set_labels = []

        for sample in test_set_ids:
            count += 1
            sample_len = int(samples_length[sample, 0])
            sample_features = data[sample:sample + sample_len, :]
            sample_label = labels[sample, 0]
            predict_label = self.predict(sample_features, records_dict, k, window_size)
            if predict_label == sample_label:
                right_count += 1
                # print(right_count)

            # break
            pre_labels.append(predict_label)
            set_labels.append(sample_label)
            print(count)
            # print(predict_label, sample_label)

        accuracy = float(right_count)/count
        end = time.time()
        print(end - start)
        print('whole_accuracy: %f' % accuracy)

        return pre_labels, set_labels

    def _cal_nearest_by_dtw(self, sample, records, topk=1, window_size=15, cal_version=1):
        # input two matrix, output a distance matrix
        # sample shape [Ts, features], records shape [records_num, Tr, features]
        if cal_version == 1:
            # version1
            dtw_distance = np.sum(records**2, axis=-1, keepdims=True) - 2*records.dot(sample.T) + \
                           np.sum(sample**2, axis=-1).T.reshape((1, 1, -1))
            for j in range(1, min(window_size+1, dtw_distance.shape[2])):
                dtw_distance[:, 0, j] += dtw_distance[:, 0, j - 1]
            for i in range(1, min(window_size+1, dtw_distance.shape[1])):
                dtw_distance[:, i, 0] += dtw_distance[:, i - 1, 0]

            for i in range(1, dtw_distance.shape[1]):
                for j in range(max(i-window_size, 1), min(dtw_distance.shape[2], i+window_size+1)):
                    temp = dtw_distance[:, i, j]
                    if i-j == window_size:
                        dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                  dtw_distance[:, i - 1, j])
                    elif j-i == window_size:
                        dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                dtw_distance[:, i, j - 1])
                    else:
                        dtw_distance[:, i, j] = temp + np.minimum(np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                             dtw_distance[:, i, j - 1]),
                                                                  dtw_distance[:, i - 1, j])

        elif cal_version == 2:
            # version 2
            dtw_distance = np.zeros((records.shape[0], records.shape[1], sample.shape[0]))

            def dot_distance(i, j):
                return np.sum((records[:, i, :]-sample[j, :]) ** 2, axis=-1)

            dtw_distance[:, 0, 0] = dot_distance(0, 0)

            for j in range(1, min(window_size+1, dtw_distance.shape[2])):
                dtw_distance[:, 0, j] = dtw_distance[:, 0, j-1] + dot_distance(0, j)
            for i in range(1, min(window_size+1, dtw_distance.shape[1])):
                dtw_distance[:, i, 0] = dtw_distance[:, i-1, 0] + dot_distance(i, 0)

            for i in range(1, dtw_distance.shape[1]):
                for j in range(max(i-window_size, 1), min(dtw_distance.shape[2], i+window_size+1)):
                    temp = dot_distance(i, j)
                    if i-j == window_size:
                        dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                  dtw_distance[:, i - 1, j])

                    elif j-i == window_size:
                        dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                  dtw_distance[:, i, j - 1])

                    else:
                        dtw_distance[:, i, j] = temp + np.minimum(np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                             dtw_distance[:, i, j - 1]),
                                                                  dtw_distance[:, i - 1, j])

        distance = dtw_distance[:, -1, -1]
        nearest_index = heapq.nsmallest(topk, range(records.shape[0]), distance.take)
        nearest_distance = distance[nearest_index]
        return nearest_index, nearest_distance  # list and 1-D vector

class Derivative_DTW:
    def __init__(self):

        return

    def train(self, training_data, labels, train_set_sample_ids, samples_length, sampling_ratio=0.001):
        # change sequence to derivative sequence
        self.training_data = training_data
        self.labels = labels
        self.train_set_sample_ids = train_set_sample_ids
        self.samples_length = samples_length

        self.record_sample_ids = random.sample(training_sample_ids, int(len(training_sample_ids)*sampling_ratio))
        # print(len(record_sample_ids))

        data_id_by_length = {}

        for i in self.record_sample_ids:
            if self.samples_length[i, 0] not in data_id_by_length:
                data_id_by_length[self.samples_length[i, 0]] = []
            data_id_by_length[self.samples_length[i, 0]].append(i)

        for each in data_id_by_length:
            sample_num = len(data_id_by_length[each])
            sample_len = int(each)
            print(str(each) + ':', sample_num)
            part_features = np.zeros((sample_num, sample_len-1, self.training_data.shape[1]))
            part_labels = np.zeros((sample_num, 1))
            for i, sample in enumerate(data_id_by_length[each]):
                part_features[i, :, :] = self.training_data[sample + 1:sample + sample_len, :] \
                                         - self.training_data[sample:sample + sample_len - 1, :]
                part_labels[i, 0] = self.labels[sample, 0]

            file_name = './data/dtw/derivative_l_%d.npz' % sample_len
            np.savez(file_name, features=part_features, labels=part_labels)

        return

    def predict(self, data_features, records_dict, k=1, window_size=15):
        # predict by all the records with different lengths on one sample
        predict_labels = []
        predict_distance = []
        for sample_len in range(16, 31):
            saved_features = records_dict[sample_len][0]
            saved_labels = records_dict[sample_len][1]
            nearest_index, nearest_distance = self._cal_nearest_by_dtw(data_features, saved_features, k, window_size,
                                                                       cal_version=1)
            nearest_labels = saved_labels[nearest_index, :].reshape((-1)).tolist()
            nearest_distance = nearest_distance.reshape((-1)).tolist()
            predict_labels.extend(nearest_labels)
            predict_distance.extend(nearest_distance)

        # print(predict_distance)
        # print(predict_labels)
        # print()
        nearest_index = heapq.nsmallest(k, range(len(predict_distance)), key=lambda x: predict_distance[x])
        count = {}
        for j in range(k):
            count[predict_labels[nearest_index[j]]] = count.get(predict_labels[nearest_index[j]], 0) + 1
        max_label = None
        max_time = 0
        for each in count:
            if count[each] > max_time:
                max_label = each
                max_time = count[each]

        return max_label

    def test(self, data, labels, test_set_ids, samples_length, k=1, window_size=15):
        # generate one data, judge once
        # save all records in memory, 10 percent occupies less than 1.44GB
        # count accuracy when judge

        records_dict = {}
        for sample_len in range(16, 31):
            saved_data = np.load('./data/dtw/derivative_l_%d.npz' % sample_len)
            saved_features = saved_data['features']
            saved_labels = saved_data['labels']
            records_dict[sample_len] = (saved_features, saved_labels)

        right_count = 0
        count = 0

        start = time.time()
        pre_labels = []
        set_labels = []

        for sample in test_set_ids:
            count += 1
            sample_len = int(samples_length[sample, 0])
            sample_features = data[sample + 1:sample + sample_len, :] - data[sample:sample + sample_len - 1, :]
            sample_label = labels[sample, 0]
            predict_label = self.predict(sample_features, records_dict, k, window_size)
            if predict_label == sample_label:
                right_count += 1
                # print(right_count)

            # break
            pre_labels.append(predict_label)
            set_labels.append(sample_label)
            print(count)
            # print(predict_label, sample_label)

        accuracy = float(right_count)/count
        end = time.time()
        print(end - start)
        print('whole_accuracy: %f' % accuracy)

        return pre_labels, set_labels

    def _cal_nearest_by_dtw(self, sample, records, topk=1, window_size=15, cal_version=1):
        # input two matrix, output a distance matrix
        # sample shape [Ts, features], records shape [records_num, Tr, features]
        if cal_version == 1:
            # version1
            dtw_distance = np.sum(records**2, axis=-1, keepdims=True) - 2*records.dot(sample.T) + \
                           np.sum(sample**2, axis=-1).T.reshape((1, 1, -1))
            for j in range(1, min(window_size+1, dtw_distance.shape[2])):
                dtw_distance[:, 0, j] += dtw_distance[:, 0, j - 1]
            for i in range(1, min(window_size+1, dtw_distance.shape[1])):
                dtw_distance[:, i, 0] += dtw_distance[:, i - 1, 0]

            for i in range(1, dtw_distance.shape[1]):
                for j in range(max(i-window_size, 1), min(dtw_distance.shape[2], i+window_size+1)):
                    temp = dtw_distance[:, i, j]
                    if i-j == window_size:
                        dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                  dtw_distance[:, i - 1, j])
                    elif j-i == window_size:
                        dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                dtw_distance[:, i, j - 1])
                    else:
                        dtw_distance[:, i, j] = temp + np.minimum(np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                             dtw_distance[:, i, j - 1]),
                                                                  dtw_distance[:, i - 1, j])

        elif cal_version == 2:
            # version 2
            dtw_distance = np.zeros((records.shape[0], records.shape[1], sample.shape[0]))

            def dot_distance(i, j):
                return np.sum((records[:, i, :]-sample[j, :]) ** 2, axis=-1)

            dtw_distance[:, 0, 0] = dot_distance(0, 0)

            for j in range(1, min(window_size+1, dtw_distance.shape[2])):
                dtw_distance[:, 0, j] = dtw_distance[:, 0, j-1] + dot_distance(0, j)
            for i in range(1, min(window_size+1, dtw_distance.shape[1])):
                dtw_distance[:, i, 0] = dtw_distance[:, i-1, 0] + dot_distance(i, 0)

            for i in range(1, dtw_distance.shape[1]):
                for j in range(max(i-window_size, 1), min(dtw_distance.shape[2], i+window_size+1)):
                    temp = dot_distance(i, j)
                    if i-j == window_size:
                        dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                  dtw_distance[:, i - 1, j])

                    elif j-i == window_size:
                        dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                  dtw_distance[:, i, j - 1])

                    else:
                        dtw_distance[:, i, j] = temp + np.minimum(np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                             dtw_distance[:, i, j - 1]),
                                                                  dtw_distance[:, i - 1, j])

        distance = dtw_distance[:, -1, -1]
        nearest_index = heapq.nsmallest(topk, range(records.shape[0]), distance.take)
        nearest_distance = distance[nearest_index]
        return nearest_index, nearest_distance  # list and 1-D vector

class Weighted_DTW:
    def __init__(self):

        return

    def train(self, training_data, labels, train_set_sample_ids, samples_length, sampling_ratio=0.001):
        self.training_data = training_data
        self.labels = labels
        self.train_set_sample_ids = train_set_sample_ids
        self.samples_length = samples_length

        self.record_sample_ids = random.sample(training_sample_ids, int(len(training_sample_ids)*sampling_ratio))
        # print(len(record_sample_ids))

        data_id_by_length = {}

        for i in self.record_sample_ids:
            if self.samples_length[i, 0] not in data_id_by_length:
                data_id_by_length[self.samples_length[i, 0]] = []
            data_id_by_length[self.samples_length[i, 0]].append(i)

        for each in data_id_by_length:
            sample_num = len(data_id_by_length[each])
            sample_len = int(each)
            print(str(each) + ':', sample_num)
            part_features = np.zeros((sample_num, sample_len, self.training_data.shape[1]))
            part_labels = np.zeros((sample_num, 1))
            for i, sample in enumerate(data_id_by_length[each]):
                part_features[i, :, :] = self.training_data[sample:sample + sample_len, :]
                part_labels[i, 0] = self.labels[sample, 0]

            file_name = './data/dtw/l_%d.npz' % sample_len
            np.savez(file_name, features=part_features, labels=part_labels)

        return

    def predict(self, data_features, records_dict, k=1, window_size=15, weight_importance=1):
        # predict by all the records with different lengths on one sample
        predict_labels = []
        predict_distance = []
        for sample_len in range(16, 31):
            saved_features = records_dict[sample_len][0]
            saved_labels = records_dict[sample_len][1]
            nearest_index, nearest_distance = self._cal_nearest_by_dtw(data_features, saved_features, k, window_size,
                                                                       weight_importance=weight_importance, cal_version=1)
            nearest_labels = saved_labels[nearest_index, :].reshape((-1)).tolist()
            nearest_distance = nearest_distance.reshape((-1)).tolist()
            predict_labels.extend(nearest_labels)
            predict_distance.extend(nearest_distance)

        # print(predict_distance)
        # print(predict_labels)
        # print()
        nearest_index = heapq.nsmallest(k, range(len(predict_distance)), key=lambda x: predict_distance[x])
        count = {}
        for j in range(k):
            count[predict_labels[nearest_index[j]]] = count.get(predict_labels[nearest_index[j]], 0) + 1
        max_label = None
        max_time = 0
        for each in count:
            if count[each] > max_time:
                max_label = each
                max_time = count[each]

        return max_label

    def test(self, data, labels, test_set_ids, samples_length, k=1, window_size=15, weight_importance=1):
        # generate one data, judge once
        # save all records in memory, 10 percent occupies less than 1.44GB
        # count accuracy when judge

        records_dict = {}
        for sample_len in range(16, 31):
            saved_data = np.load('./data/dtw/l_%d.npz' % sample_len)
            saved_features = saved_data['features']
            saved_labels = saved_data['labels']
            records_dict[sample_len] = (saved_features, saved_labels)

        right_count = 0
        count = 0

        start = time.time()

        pre_labels = []
        set_labels = []

        for sample in test_set_ids:
            count += 1
            sample_len = int(samples_length[sample, 0])
            sample_features = data[sample:sample + sample_len, :]
            sample_label = labels[sample, 0]
            predict_label = self.predict(sample_features, records_dict, k, window_size, weight_importance)
            if predict_label == sample_label:
                right_count += 1
                # print(right_count)

            # break
            pre_labels.append(predict_label)
            set_labels.append(sample_label)
            print(count)
            # print(predict_label, sample_label)

        accuracy = float(right_count)/count
        end = time.time()
        print(end - start)
        print('whole_accuracy: %f' % accuracy)

        return pre_labels, set_labels

    def _weight_cal(self, a, penalty, sequence_len):
        w_max = 1
        result = w_max/(1 + np.exp(-penalty*(a - sequence_len/2)))
        return result

    def _cal_nearest_by_dtw(self, sample, records, topk=1, window_size=15, weight_importance=1, cal_version=1):
        # input two matrix, output a distance matrix
        # sample shape [Ts, features], records shape [records_num, Tr, features]
        if cal_version == 1:
            # version1
            dtw_distance = np.sum(records**2, axis=-1, keepdims=True) - 2*records.dot(sample.T) + \
                           np.sum(sample**2, axis=-1).T.reshape((1, 1, -1))

            # weight calculation
            vector1 = np.array(range(dtw_distance.shape[1])).reshape((-1, 1))
            vector2 = np.array(range(dtw_distance.shape[2])).reshape((1, -1))
            ij = np.abs(vector1-vector2)
            sequence_len = min(dtw_distance.shape[1], dtw_distance.shape[2])
            weight = self._weight_cal(ij, weight_importance, sequence_len)
            weight = weight.reshape((1, weight.shape[0], weight.shape[1]))
            dtw_distance = weight * dtw_distance

            for j in range(1, min(window_size+1, dtw_distance.shape[2])):
                dtw_distance[:, 0, j] += dtw_distance[:, 0, j - 1]
            for i in range(1, min(window_size+1, dtw_distance.shape[1])):
                dtw_distance[:, i, 0] += dtw_distance[:, i - 1, 0]

            for i in range(1, dtw_distance.shape[1]):
                for j in range(max(i-window_size, 1), min(dtw_distance.shape[2], i+window_size+1)):
                    temp = dtw_distance[:, i, j]
                    if i-j == window_size:
                        dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                  dtw_distance[:, i - 1, j])
                    elif j-i == window_size:
                        dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                dtw_distance[:, i, j - 1])
                    else:
                        dtw_distance[:, i, j] = temp + np.minimum(np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                             dtw_distance[:, i, j - 1]),
                                                                  dtw_distance[:, i - 1, j])

        elif cal_version == 2:
            # version 2
            dtw_distance = np.zeros((records.shape[0], records.shape[1], sample.shape[0]))

            def dot_distance(i, j):
                return np.sum((records[:, i, :]-sample[j, :]) ** 2, axis=-1)

            dtw_distance[:, 0, 0] = dot_distance(0, 0)

            for j in range(1, min(window_size+1, dtw_distance.shape[2])):
                dtw_distance[:, 0, j] = dtw_distance[:, 0, j-1] + dot_distance(0, j)
            for i in range(1, min(window_size+1, dtw_distance.shape[1])):
                dtw_distance[:, i, 0] = dtw_distance[:, i-1, 0] + dot_distance(i, 0)

            for i in range(1, dtw_distance.shape[1]):
                for j in range(max(i-window_size, 1), min(dtw_distance.shape[2], i+window_size+1)):
                    temp = dot_distance(i, j)
                    if i-j == window_size:
                        dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                  dtw_distance[:, i - 1, j])

                    elif j-i == window_size:
                        dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                  dtw_distance[:, i, j - 1])

                    else:
                        dtw_distance[:, i, j] = temp + np.minimum(np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                             dtw_distance[:, i, j - 1]),
                                                                  dtw_distance[:, i - 1, j])

        distance = dtw_distance[:, -1, -1]
        nearest_index = heapq.nsmallest(topk, range(records.shape[0]), distance.take)
        nearest_distance = distance[nearest_index]
        return nearest_index, nearest_distance  # list and 1-D vector

class Weighted_Derivative_DTW:
    def __init__(self):

        return

    def train(self, training_data, labels, train_set_sample_ids, samples_length, sampling_ratio=0.001):
        # change sequence to derivative sequence
        self.training_data = training_data
        self.labels = labels
        self.train_set_sample_ids = train_set_sample_ids
        self.samples_length = samples_length

        self.record_sample_ids = random.sample(training_sample_ids, int(len(training_sample_ids)*sampling_ratio))
        # print(len(record_sample_ids))

        data_id_by_length = {}

        for i in self.record_sample_ids:
            if self.samples_length[i, 0] not in data_id_by_length:
                data_id_by_length[self.samples_length[i, 0]] = []
            data_id_by_length[self.samples_length[i, 0]].append(i)

        for each in data_id_by_length:
            sample_num = len(data_id_by_length[each])
            sample_len = int(each)
            print(str(each) + ':', sample_num)
            part_features = np.zeros((sample_num, sample_len-1, self.training_data.shape[1]))
            part_labels = np.zeros((sample_num, 1))
            for i, sample in enumerate(data_id_by_length[each]):
                part_features[i, :, :] = self.training_data[sample + 1:sample + sample_len, :] \
                                         - self.training_data[sample:sample + sample_len - 1, :]
                part_labels[i, 0] = self.labels[sample, 0]

            file_name = './data/dtw/derivative_l_%d.npz' % sample_len
            np.savez(file_name, features=part_features, labels=part_labels)

        return

    def predict(self, data_features, records_dict, k=1, window_size=15, weight_importance=1):
        # predict by all the records with different lengths on one sample
        predict_labels = []
        predict_distance = []
        for sample_len in range(16, 31):
            saved_features = records_dict[sample_len][0]
            saved_labels = records_dict[sample_len][1]
            nearest_index, nearest_distance = self._cal_nearest_by_dtw(data_features, saved_features, k, window_size,
                                                                       weight_importance=weight_importance, cal_version=1)
            nearest_labels = saved_labels[nearest_index, :].reshape((-1)).tolist()
            nearest_distance = nearest_distance.reshape((-1)).tolist()
            predict_labels.extend(nearest_labels)
            predict_distance.extend(nearest_distance)

        # print(predict_distance)
        # print(predict_labels)
        # print()
        nearest_index = heapq.nsmallest(k, range(len(predict_distance)), key=lambda x: predict_distance[x])
        count = {}
        for j in range(k):
            count[predict_labels[nearest_index[j]]] = count.get(predict_labels[nearest_index[j]], 0) + 1
        max_label = None
        max_time = 0
        for each in count:
            if count[each] > max_time:
                max_label = each
                max_time = count[each]

        return max_label

    def test(self, data, labels, test_set_ids, samples_length, k=1, window_size=15, weight_importance=1):
        # generate one data, judge once
        # save all records in memory, 10 percent occupies less than 1.44GB
        # count accuracy when judge

        records_dict = {}
        for sample_len in range(16, 31):
            saved_data = np.load('./data/dtw/derivative_l_%d.npz' % sample_len)
            saved_features = saved_data['features']
            saved_labels = saved_data['labels']
            records_dict[sample_len] = (saved_features, saved_labels)

        right_count = 0
        count = 0

        start = time.time()

        pre_labels = []
        set_labels = []
        for sample in test_set_ids:
            count += 1
            sample_len = int(samples_length[sample, 0])
            sample_features = data[sample + 1:sample + sample_len, :] - data[sample:sample + sample_len - 1, :]
            sample_label = labels[sample, 0]
            predict_label = self.predict(sample_features, records_dict, k, window_size, weight_importance)
            if predict_label == sample_label:
                right_count += 1
                # print(right_count)

            pre_labels.append(predict_label)
            set_labels.append(sample_label)
            # break
            print(count)
            # print(predict_label, sample_label)

        accuracy = float(right_count)/count
        end = time.time()
        print(end - start)
        print('whole_accuracy: %f' % accuracy)

        return pre_labels, set_labels

    def _weight_cal(self, a, penalty, sequence_len):
        w_max = 1
        result = w_max/(1 + np.exp(-penalty*(a - sequence_len/2)))
        return result

    def _cal_nearest_by_dtw(self, sample, records, topk=1, window_size=15, weight_importance=1, cal_version=1):
        # input two matrix, output a distance matrix
        # sample shape [Ts, features], records shape [records_num, Tr, features]
        if cal_version == 1:
            # version1
            dtw_distance = np.sum(records**2, axis=-1, keepdims=True) - 2*records.dot(sample.T) + \
                           np.sum(sample**2, axis=-1).T.reshape((1, 1, -1))

            # weight calculation
            vector1 = np.array(range(dtw_distance.shape[1])).reshape((-1, 1))
            vector2 = np.array(range(dtw_distance.shape[2])).reshape((1, -1))
            ij = np.abs(vector1-vector2)
            sequence_len = min(dtw_distance.shape[1], dtw_distance.shape[2])
            weight = self._weight_cal(ij, weight_importance, sequence_len)
            weight = weight.reshape((1, weight.shape[0], weight.shape[1]))
            dtw_distance = weight * dtw_distance

            for j in range(1, min(window_size+1, dtw_distance.shape[2])):
                dtw_distance[:, 0, j] += dtw_distance[:, 0, j - 1]
            for i in range(1, min(window_size+1, dtw_distance.shape[1])):
                dtw_distance[:, i, 0] += dtw_distance[:, i - 1, 0]

            for i in range(1, dtw_distance.shape[1]):
                for j in range(max(i-window_size, 1), min(dtw_distance.shape[2], i+window_size+1)):
                    temp = dtw_distance[:, i, j]
                    if i-j == window_size:
                        dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                  dtw_distance[:, i - 1, j])
                    elif j-i == window_size:
                        dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                dtw_distance[:, i, j - 1])
                    else:
                        dtw_distance[:, i, j] = temp + np.minimum(np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                             dtw_distance[:, i, j - 1]),
                                                                  dtw_distance[:, i - 1, j])

        elif cal_version == 2:
            # version 2
            dtw_distance = np.zeros((records.shape[0], records.shape[1], sample.shape[0]))

            def dot_distance(i, j):
                return np.sum((records[:, i, :]-sample[j, :]) ** 2, axis=-1)

            dtw_distance[:, 0, 0] = dot_distance(0, 0)

            for j in range(1, min(window_size+1, dtw_distance.shape[2])):
                dtw_distance[:, 0, j] = dtw_distance[:, 0, j-1] + dot_distance(0, j)
            for i in range(1, min(window_size+1, dtw_distance.shape[1])):
                dtw_distance[:, i, 0] = dtw_distance[:, i-1, 0] + dot_distance(i, 0)

            for i in range(1, dtw_distance.shape[1]):
                for j in range(max(i-window_size, 1), min(dtw_distance.shape[2], i+window_size+1)):
                    temp = dot_distance(i, j)
                    if i-j == window_size:
                        dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                  dtw_distance[:, i - 1, j])

                    elif j-i == window_size:
                        dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                  dtw_distance[:, i, j - 1])

                    else:
                        dtw_distance[:, i, j] = temp + np.minimum(np.minimum(dtw_distance[:, i - 1, j - 1],
                                                                             dtw_distance[:, i, j - 1]),
                                                                  dtw_distance[:, i - 1, j])

        distance = dtw_distance[:, -1, -1]
        nearest_index = heapq.nsmallest(topk, range(records.shape[0]), distance.take)
        nearest_distance = distance[nearest_index]
        return nearest_index, nearest_distance  # list and 1-D vector

class TWE:
    def __init__(self):

        return

    def train(self, training_data, labels, train_set_sample_ids, samples_length, sampling_ratio=0.001):
        self.training_data = training_data
        self.labels = labels
        self.train_set_sample_ids = train_set_sample_ids
        self.samples_length = samples_length

        self.record_sample_ids = random.sample(training_sample_ids, int(len(training_sample_ids) * sampling_ratio))
        # print(len(record_sample_ids))

        data_id_by_length = {}

        for i in self.record_sample_ids:
            if self.samples_length[i, 0] not in data_id_by_length:
                data_id_by_length[self.samples_length[i, 0]] = []
            data_id_by_length[self.samples_length[i, 0]].append(i)

        for each in data_id_by_length:
            sample_num = len(data_id_by_length[each])
            sample_len = int(each)
            print(str(each) + ':', sample_num)
            part_features = np.zeros((sample_num, sample_len, self.training_data.shape[1]))
            part_labels = np.zeros((sample_num, 1))
            for i, sample in enumerate(data_id_by_length[each]):
                part_features[i, :, :] = self.training_data[sample:sample + sample_len, :]
                part_labels[i, 0] = self.labels[sample, 0]

            file_name = './data/dtw/l_%d.npz' % sample_len
            np.savez(file_name, features=part_features, labels=part_labels)

        return

    def predict(self, data_features, records_dict, k=1, stiffness=1, penalty=1):
        # predict by all the records with different lengths on one sample
        predict_labels = []
        predict_distance = []
        for sample_len in range(16, 31):
            saved_features = records_dict[sample_len][0]
            saved_labels = records_dict[sample_len][1]
            nearest_index, nearest_distance = self._cal_nearest_by_twe(data_features, saved_features, k, stiffness,
                                                                       penalty)
            nearest_labels = saved_labels[nearest_index, :].reshape((-1)).tolist()
            nearest_distance = nearest_distance.reshape((-1)).tolist()
            predict_labels.extend(nearest_labels)
            predict_distance.extend(nearest_distance)

        # print(predict_distance)
        # print(predict_labels)
        # print()
        nearest_index = heapq.nsmallest(k, range(len(predict_distance)), key=lambda x: predict_distance[x])
        count = {}
        for j in range(k):
            count[predict_labels[nearest_index[j]]] = count.get(predict_labels[nearest_index[j]], 0) + 1
        max_label = None
        max_time = 0
        for each in count:
            if count[each] > max_time:
                max_label = each
                max_time = count[each]

        return max_label

    def test(self, data, labels, test_set_ids, samples_length, k=1, stiffness=1, penalty=1):
        # generate one data, judge once
        # save all records in memory, 10 percent occupies less than 1.44GB
        # count accuracy when judge

        records_dict = {}
        for sample_len in range(16, 31):
            saved_data = np.load('./data/dtw/l_%d.npz' % sample_len)
            saved_features = saved_data['features']
            saved_labels = saved_data['labels']
            records_dict[sample_len] = (saved_features, saved_labels)

        right_count = 0
        count = 0

        start = time.time()

        pre_labels = []
        set_labels = []

        for sample in test_set_ids:
            count += 1
            sample_len = int(samples_length[sample, 0])
            sample_features = data[sample:sample + sample_len, :]
            sample_label = labels[sample, 0]
            predict_label = self.predict(sample_features, records_dict, k, stiffness, penalty)
            if predict_label == sample_label:
                right_count += 1
                # print(right_count)

            # break
            pre_labels.append(predict_label)
            set_labels.append(sample_label)
            print(count)
            # print(predict_label, sample_label)

        accuracy = float(right_count) / count
        end = time.time()
        print(end - start)
        print('whole_accuracy: %f' % accuracy)

        return pre_labels, set_labels

    def _cal_nearest_by_twe(self, sample, records, topk=1, stiffness=1, penalty=1, cal_version=1):
        # todo finish twe, erase window_size in other function
        # input two matrix, output a distance matrix
        # sample shape [Ts, features], records shape [records_num, Tr, features]
        if cal_version == 1:
            # version1
            edit_dist_records = np.sum((records[:, :-1, :] - records[:, 1:, :]) ** 2, axis=-1)
            edit_dist_sample = np.sum((sample[:-1, :] - sample[1:, :]) ** 2, axis=-1)

            ij_distance = np.sum(records ** 2, axis=-1, keepdims=True) - 2 * records.dot(sample.T) + \
                           np.sum(sample ** 2, axis=-1).T.reshape((1, 1, -1))

            twe_distance = np.zeros((records.shape[0], records.shape[1]+1, sample.shape[0]))

            twe_distance[:, 1, 0] = np.sum(records[:, 0, :] ** 2, axis=-1)
            twe_distance[:, 0, 1] = np.sum(sample[0, :] ** 2, axis=-1)

            for i in range(2, twe_distance.shape[1]):
                twe_distance[:, i, 0] = twe_distance[:, i - 1, 0] + edit_dist_records[:, i - 2]

            for j in range(2, twe_distance.shape[2]):
                twe_distance[:, 0, j] = twe_distance[:, 0, j - 1] + edit_dist_sample[j - 2]

            for i in range(1, twe_distance.shape[1]):
                for j in range(1, twe_distance.shape[2]):
                    if i > 1 and j > 1:
                        dist1 = twe_distance[:, i - 1, j - 1] + stiffness * abs(i-j) * 2 \
                                + ij_distance[:, i - 1, j - 1] + ij_distance[:, i - 2, j - 2]
                    else:
                        dist1 = twe_distance[:, i - 1, j - 1] + stiffness * abs(i-j) \
                                + ij_distance[:, i - 1, j - 1]
                    if i > 1:
                        dist2 = twe_distance[:, i - 1, j] + penalty + stiffness \
                                + edit_dist_records[:, i - 2]
                    else:
                        dist2 = twe_distance[:, i - 1, j] + twe_distance[:, 1, 0] + penalty

                    if j > 1:
                        dist3 = twe_distance[:, i, j - 1] + penalty + stiffness \
                                + edit_dist_sample[j - 2]
                    else:
                        dist3 = twe_distance[:, i, j - 1] + twe_distance[:, 0, 1] + penalty
                    twe_distance[:, i, j] = np.minimum(np.minimum(dist1, dist2), dist3)

        elif cal_version == 2:
            pass
            # version 2
            # dtw_distance = np.zeros((records.shape[0], records.shape[1], sample.shape[0]))
            #
            # def dot_distance(i, j):
            #     return np.sum((records[:, i, :] - sample[j, :]) ** 2, axis=-1)
            #
            # dtw_distance[:, 0, 0] = dot_distance(0, 0)
            #
            # for j in range(1, min(window_size + 1, dtw_distance.shape[2])):
            #     dtw_distance[:, 0, j] = dtw_distance[:, 0, j - 1] + dot_distance(0, j)
            # for i in range(1, min(window_size + 1, dtw_distance.shape[1])):
            #     dtw_distance[:, i, 0] = dtw_distance[:, i - 1, 0] + dot_distance(i, 0)
            #
            # for i in range(1, dtw_distance.shape[1]):
            #     for j in range(max(i - window_size, 1), min(dtw_distance.shape[2], i + window_size + 1)):
            #         temp = dot_distance(i, j)
            #         if i - j == window_size:
            #             dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
            #                                                       dtw_distance[:, i - 1, j])
            #
            #         elif j - i == window_size:
            #             dtw_distance[:, i, j] = temp + np.minimum(dtw_distance[:, i - 1, j - 1],
            #                                                       dtw_distance[:, i, j - 1])
            #
            #         else:
            #             dtw_distance[:, i, j] = temp + np.minimum(np.minimum(dtw_distance[:, i - 1, j - 1],
            #                                                                  dtw_distance[:, i, j - 1]),
            #                                                       dtw_distance[:, i - 1, j])

        distance = twe_distance[:, -1, -1]
        nearest_index = heapq.nsmallest(topk, range(records.shape[0]), distance.take)
        nearest_distance = distance[nearest_index]
        return nearest_index, nearest_distance  # list and 1-D vector

class Elastic_Ensemble:
    def __init__(self):
        self.dtw = DTW()
        self.ddtw = Derivative_DTW()
        self.wdtw = Weighted_DTW()
        self.wddtw = Weighted_Derivative_DTW()
        self.twe = TWE()
        return

    def train(self, training_data, labels, train_set_sample_ids, samples_length, sampling_ratio=0.001):
        self.training_data = training_data
        self.labels = labels
        self.train_set_sample_ids = train_set_sample_ids
        self.samples_length = samples_length

        self.record_sample_ids = random.sample(training_sample_ids, int(len(training_sample_ids)*sampling_ratio))
        # print(len(record_sample_ids))

        data_id_by_length = {}

        for i in self.record_sample_ids:
            if self.samples_length[i, 0] not in data_id_by_length:
                data_id_by_length[self.samples_length[i, 0]] = []
            data_id_by_length[self.samples_length[i, 0]].append(i)

        for each in data_id_by_length:
            sample_num = len(data_id_by_length[each])
            sample_len = int(each)
            print(str(each) + ':', sample_num)
            part_features = np.zeros((sample_num, sample_len, self.training_data.shape[1]))
            part_labels = np.zeros((sample_num, 1))
            for i, sample in enumerate(data_id_by_length[each]):
                part_features[i, :, :] = self.training_data[sample:sample + sample_len, :]
                part_labels[i, 0] = self.labels[sample, 0]

            file_name = './data/dtw/l_%d.npz' % sample_len
            np.savez(file_name, features=part_features, labels=part_labels)

        for each in data_id_by_length:
            sample_num = len(data_id_by_length[each])
            sample_len = int(each)
            print(str(each) + ':', sample_num)
            part_features = np.zeros((sample_num, sample_len-1, self.training_data.shape[1]))
            part_labels = np.zeros((sample_num, 1))
            for i, sample in enumerate(data_id_by_length[each]):
                part_features[i, :, :] = self.training_data[sample + 1:sample + sample_len, :] \
                                         - self.training_data[sample:sample + sample_len - 1, :]
                part_labels[i, 0] = self.labels[sample, 0]

            file_name = './data/dtw/derivative_l_%d.npz' % sample_len
            np.savez(file_name, features=part_features, labels=part_labels)

        return

    def test(self, data, labels, test_set_ids, samples_length, predictor_weight=None, k=1, window_size=15, weight_importance=1, stiffness=1, penalty=1):
        # generate one data, judge once
        # save all records in memory, 10 percent occupies less than 1.44GB
        # count accuracy when judge
        if predictor_weight is None:
            predictor_weight = [1, 1, 1, 1, 1]

        records_dict = {}
        for sample_len in range(16, 31):
            saved_data = np.load('./data/dtw/l_%d.npz' % sample_len)
            saved_features = saved_data['features']
            saved_labels = saved_data['labels']
            records_dict[sample_len] = (saved_features, saved_labels)

        records_d_dict = {}
        for sample_len in range(16, 31):
            saved_data = np.load('./data/dtw/derivative_l_%d.npz' % sample_len)
            saved_features = saved_data['features']
            saved_labels = saved_data['labels']
            records_d_dict[sample_len] = (saved_features, saved_labels)

        right_count = 0
        count = 0

        start = time.time()

        pre_labels = []
        set_labels = []

        for sample in test_set_ids:
            count += 1
            for_vote_labels = []
            sample_len = int(samples_length[sample, 0])
            sample_features = data[sample:sample + sample_len, :]
            sample_label = labels[sample, 0]

            for_vote_labels.append(self.dtw.predict(sample_features, records_dict, k, window_size))
            for_vote_labels.append(self.ddtw.predict(sample_features, records_d_dict, k, window_size))
            for_vote_labels.append(self.wdtw.predict(sample_features, records_dict, k, window_size, weight_importance))
            for_vote_labels.append(self.wddtw.predict(sample_features, records_d_dict, k, window_size, weight_importance))
            for_vote_labels.append(self.twe.predict(sample_features, records_dict, k, stiffness, penalty))

            print(for_vote_labels, sample_label)
            label_vote_count = {}
            for i in range(len(for_vote_labels)):
                label = for_vote_labels[i]
                label_vote_count[label] = label_vote_count.get(label, 0) + predictor_weight[i]
            # print(label_vote_count)

            predict_label = None
            max_time = 0
            for each in label_vote_count:
                if label_vote_count[each] > max_time:
                    predict_label = each
                    max_time = label_vote_count[each]
            # print(predict_label)

            if predict_label == sample_label:
                right_count += 1
                # print(right_count)

            # break
            pre_labels.append(predict_label)
            set_labels.append(sample_label)
            print(count)
            # print(predict_label, sample_label)

        accuracy = float(right_count)/count
        end = time.time()
        print(end - start)
        print('whole_accuracy: %f' % accuracy)

        return pre_labels, set_labels


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

    # generate training samples' id

    with open(common_para['path']['data_set_ids_file'], 'r') as f:
        data_set_ids = json.load(f)
    training_sample_ids = data_set_ids['training_set']
    test_sample_ids = data_set_ids['test_set']

    # print('dtw')
    # dtw = DTW()
    # # dtw.train(data['features'], data['labels'], training_sample_ids, data['samples_length'], sampling_ratio=0.03)
    # # dtw.test(data['features'], data['labels'], test_sample_ids[:100], data['samples_length'], k=1, window_size=15)
    # dtw.test(data['features'], data['labels'], test_sample_ids[:100], data['samples_length'], k=6, window_size=15)
    #
    # print('ddtw')
    # ddtw = Derivative_DTW()
    # # ddtw.train(data['features'], data['labels'], training_sample_ids, data['samples_length'], sampling_ratio=0.03)
    # ddtw.test(data['features'], data['labels'], test_sample_ids[:100], data['samples_length'], k=5, window_size=15)

    print('wdtw')
    wdtw = Weighted_DTW()
    start = time.time()
    y_predict, y_true = wdtw.test(data['features'], data['labels'], test_sample_ids[:10000], data['samples_length'], k=5, window_size=15,
              weight_importance=4.5)  # alternative weight importance
    end = time.time()
    print('Time cost:', end-start)

    y_predict = np.array(y_predict)
    y_true = np.array(y_true)

    np.savez('./data/confusion_matrix/dtw/dtw_predict.npz', y_predict=y_predict, y_true=y_true)

    # print('wddtw')
    # wddtw = Weighted_Derivative_DTW()
    # wddtw.test(data['features'], data['labels'], test_sample_ids[:100], data['samples_length'], k=5, window_size=15,
    #           weight_importance=20)  # alternative weight importance []
    #
    # print('twe')
    # twe = TWE()
    # twe.test(data['features'], data['labels'], test_sample_ids[:100], data['samples_length'], k=5,
    #          stiffness=4, penalty=2)

    # ee = Elastic_Ensemble()
    # ee.test(data['features'], data['labels'], test_sample_ids[:10], data['samples_length'], predictor_weight=[1, 1, 3, 1, 1],
    #         k=5, window_size=15, weight_importance=6.4, stiffness=4, penalty=2)