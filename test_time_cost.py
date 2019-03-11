import numpy as np
import BPNN_model, KNN_sequence_Model
import RNN_models, CNN_models, AttentionModel
import configparser
import json
import time


def rnn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, cell_type, random_seed=None):
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('%s_model time test start' % cell_type)

    model = RNN_models.FixedLengthRNN(sequence_fix_length, data['features'].shape[1],
                                     class_num=class_num, cell_type=cell_type)

    model.load_model('./model/%s_model_v2' % cell_type)

    whole_time_on_test = model.test_time(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    whole_time_on_test_str = '%s time cost on predicting %d test samples: %fs\n' % (cell_type,
                                                                                    len(test_sample_ids),
                                                                                    whole_time_on_test)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # print('cost time %f' % (end_time - start_time))
    print('%s_model test over\n' % cell_type)
    return whole_time_on_test_str


def cnn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, random_seed=None):
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('cnn_model time test start')

    cnn = CNN_models.ConvSequence2One(sequence_fix_length, data['features'].shape[1],
                                      class_num=class_num)

    cnn.load_model('./model/cnn_model_v2')

    whole_time_on_test = cnn.test_time(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    whole_time_on_test_str = 'cnn time cost on predicting %d test samples: %fs\n' % (len(test_sample_ids),
                                                                                     whole_time_on_test)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # print('cost time %f' % (end_time - start_time))
    print('cnn_model test over\n')
    return whole_time_on_test_str


def atn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, random_seed=None):
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('atn_model time test start')

    atn = AttentionModel.OnlyAttention(sequence_fix_length, data['features'].shape[1], class_num=class_num,
                                       network_hyperparameters='./data/attention_network_hyperparameters_v2.json')

    atn.load_model('./model/atn_model_v2')

    whole_time_on_test = atn.test_time(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    whole_time_on_test_str = 'sum_a_cnn time cost on predicting %d test samples: %fs\n' % (len(test_sample_ids),
                                                                                     whole_time_on_test)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # print('cost time %f' % (end_time - start_time))
    print('atn_model test over\n')
    return whole_time_on_test_str


def bpnn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, random_seed=None):
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('bpnn_model test start')

    bpnn = BPNN_model.BPNN(sequence_fix_length, data['features'].shape[1], class_num=class_num)

    bpnn.load_model('./model/bpnn_model_v2')

    whole_time_on_test = bpnn.test_time(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    whole_time_on_test_str = 'bpnn time cost on predicting %d test samples: %fs\n' % (len(test_sample_ids),
                                                                                     whole_time_on_test)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # print('cost time %f' % (end_time - start_time))
    print('bpnn_model test over\n')
    return whole_time_on_test_str

def knn_exams(data, training_sample_ids, test_sample_ids):
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('knn_model test start')
    start_time = time.time()
    knn = KNN_sequence_Model.KNN_Sequence()
    # knn.train(data['features'], data['labels'], training_sample_ids, data['samples_length'])
    knn.test_cpu(data['features'], data['labels'], test_sample_ids[-10000:], data['samples_length'], 100)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    end_time = time.time()
    print('cost time %f' % (end_time - start_time))
    print('knn_model test over\n')
    return


if __name__ == '__main__':
    common_para = configparser.ConfigParser()
    common_para.read('common_para.ini')
    sequence_fix_length = common_para['common_parameters'].getint('sequence_fix_length')
    foresight_steps = common_para['common_parameters'].getint('foresight_steps')
    class_num = common_para['common_parameters'].getint('class_num')
    random_seed = common_para['common_parameters'].getint('random_seed')

    data = np.load(common_para['path']['data_file'])

    with open(common_para['path']['data_set_ids_file'], 'r') as f:
        data_set_ids = json.load(f)
    training_sample_ids = data_set_ids['training_set']
    test_sample_ids = data_set_ids['test_set']

    time_str = ''
    # rnn exams
    # lstm
    tstr = rnn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, 'lstm', random_seed)
    time_str += tstr

    # gru
    tstr = rnn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, 'gru', random_seed)
    time_str += tstr

    # sru
    tstr = rnn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, 'sru', random_seed)
    time_str += tstr

    # cnn exams
    tstr = cnn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, random_seed)
    time_str += tstr

    # attention net exams
    tstr = atn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, random_seed)
    time_str += tstr

    # traditional ways
    # BPNN exams
    tstr = bpnn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, random_seed)
    time_str += tstr

    with open('./data/time_cost.txt', 'w+') as file:
        file.write(time_str)

    # # KNN exams
    # knn_exams(data, training_sample_ids, test_sample_ids)