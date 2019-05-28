import numpy as np
import BPNN_model, KNN_sequence_Model
import RNN_models, CNN_models, AttentionModel, Incremental_Learning_models
import configparser
import json
import time


def rnn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, cell_type, random_seed=None):
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('%s_model test start' % cell_type)

    model = RNN_models.FixedLengthRNN(sequence_fix_length, data['features'].shape[1],
                                     class_num=class_num, cell_type=cell_type)
    # lstm.train(data['features'], data['labels'], 1, 1024, training_sample_ids,
    #            foresight_steps=foresight_steps, reset_flag=True)
    # lstm.test(data['features'], data['labels'], test_sample_ids)
    # lstm.save_model('./model/rnn_model')
    start_time = time.time()
    model.train_v2(data['features'], data['labels'], data['samples_length'], 1, 1024, training_sample_ids,
                   test_sample_ids, foresight_steps=0, reset_flag=True, record_flag=False, random_seed=random_seed)
    end_time = time.time()
    print('cost training time %f' % (end_time - start_time))
    model.test_v2(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    # model.save_model('./model/%s_model_v2' % cell_type)

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
    print('cnn_model test start')

    cnn = CNN_models.ConvSequence2One(sequence_fix_length, data['features'].shape[1],
                                      class_num=class_num)
    # cnn.train(data['features'], data['labels'], 1, 1024, training_sample_ids)
    # cnn.test(data['features'], data['labels'], test_sample_ids)
    # cnn.save_model('./model/cnn_model')
    start_time = time.time()
    cnn.train_v2(data['features'], data['labels'], data['samples_length'], 1, 1024, training_sample_ids, test_sample_ids,
                 foresight_steps=0, reset_flag=True, record_flag=False, random_seed=random_seed)
    end_time = time.time()
    print('cost training time %f' % (end_time - start_time))
    cnn.test_v2(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    # cnn.save_model('./model/cnn_model_v2')

    whole_time_on_test = cnn.test_time(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    whole_time_on_test_str = 'cnn time cost on predicting %d test samples: %fs\n' % (len(test_sample_ids),
                                                                                     whole_time_on_test)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # print('cost time %f' % (end_time - start_time))
    print('cnn_model test over\n')
    return whole_time_on_test_str


def atn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, random_seed=None):
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('atn_model test start')

    atn = AttentionModel.CNN_Attention(sequence_fix_length, data['features'].shape[1], class_num=class_num,
                                       network_hyperparameters='./data/attention_network_hyperparameters_v2.json')

    # atn.train(data['features'], data['labels'], 1, 1024, training_sample_ids)
    # atn.test(data['features'], data['labels'], test_sample_ids)
    # atn.save_model('./model/atn_model_new')
    start_time = time.time()
    atn.train_v2(data['features'], data['labels'], data['samples_length'], 1, 1024, training_sample_ids, test_sample_ids,
                 foresight_steps=0, reset_flag=True, record_flag=False, random_seed=random_seed)
    end_time = time.time()
    print('cost training time %f' % (end_time - start_time))
    atn.test_v2(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    # atn.save_model('./model/atn_model_v2')

    whole_time_on_test = atn.test_time(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    whole_time_on_test_str = 'sum_a_cnn time cost on predicting %d test samples: %fs\n' % (len(test_sample_ids),
                                                                                     whole_time_on_test)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # print('cost time %f' % (end_time - start_time))
    print('atn_model test over\n')
    return whole_time_on_test_str

def incremental_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, incremental_sample_ids, random_seed=None):
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('incremental_model test start')

    incre_cnn_a = Incremental_Learning_models.Incremental_CNN_Attention(sequence_fix_length, data['features'].shape[1],
                                                                        class_num=class_num,
                                       network_hyperparameters='./data/attention_network_hyperparameters_v2.json',
                                       incremental_net_hyperparameters='./data/Incremental_CNN_A.json')

    incre_cnn_a.incremental_simulation(data['features'], data['labels'], data['samples_length'], 2, 1024,
                                       training_sample_ids, test_sample_ids, incremental_sample_ids,
                                       foresight_steps=0, reset_flag=True, record_flag=False, random_seed=random_seed)

    # atn.train(data['features'], data['labels'], 1, 1024, training_sample_ids)
    # atn.test(data['features'], data['labels'], test_sample_ids)
    # atn.save_model('./model/atn_model_new')
    # start_time = time.time()
    # incre_cnn_a.train_v2(data['features'], data['labels'], data['samples_length'], 1, 1024, training_sample_ids, test_sample_ids,
    #              foresight_steps=0, reset_flag=True, record_flag=True, random_seed=random_seed)
    # end_time = time.time()
    # print('cost training time %f' % (end_time - start_time))
    # incre_cnn_a.test_v2(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    # incre_cnn_a.save_model('./model/atn_model_v2')
    #
    # whole_time_on_test = incre_cnn_a.test_time(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    # whole_time_on_test_str = 'sum_a_cnn time cost on predicting %d test samples: %fs\n' % (len(test_sample_ids),
    #                                                                                  whole_time_on_test)
    #
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # # print('cost time %f' % (end_time - start_time))
    # print('atn_model test over\n')
    # return whole_time_on_test_str

def bpnn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, random_seed=None):
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('bpnn_model test start')

    bpnn = BPNN_model.BPNN(sequence_fix_length, data['features'].shape[1], class_num=class_num)

    # bpnn.train(data['features'], data['labels'], 1, 4096, training_sample_ids,
    #            foresight_steps=foresight_steps, reset_flag=True)
    # bpnn.test(data['features'], data['labels'], test_sample_ids)
    # bpnn.save_model('./model/bpnn_model')
    start_time = time.time()
    bpnn.train_v2(data['features'], data['labels'], data['samples_length'], 1, 1024, training_sample_ids, test_sample_ids,
                  foresight_steps=0, reset_flag=True, record_flag=False, random_seed=random_seed)
    end_time = time.time()
    print('cost training time %f' % (end_time - start_time))
    bpnn.test_v2(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    # bpnn.save_model('./model/bpnn_model_v2')

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
    # print(data['features'])
    # print(data['labels'])
    # print(data['samples_length'])
    # print(data['features'].shape)
    # print(data['labels'].shape)
    # print(data['samples_length'].shape)

    ### generate training samples' id
    # sample_num = data['labels'].shape[0]
    # sample_ids = [i for i in range(sample_num)]
    # training_sample_ids = random.sample(sample_ids, int(0.7*sample_num))
    # test_sample_ids = list(set(sample_ids) - set(training_sample_ids))
    # training_sample_ids = [i for i in range(10000)]
    # test_sample_ids = [i for i in range(1000, 2000)]
    with open(common_para['path']['data_set_ids_file'], 'r') as f:
        data_set_ids = json.load(f)
    training_sample_ids = data_set_ids['training_set']
    test_sample_ids = data_set_ids['test_set']
    # training_sample_ids = list(map(int, data_set_ids['training_set']))
    # test_sample_ids = list(map(int, data_set_ids['test_set']))
    # print(type(training_sample_ids), type(test_sample_ids))

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
    #
    # # attention net exams
    tstr = atn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, random_seed)
    time_str += tstr
    #
    # traditional ways
    # BPNN exams
    tstr = bpnn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, random_seed)
    time_str += tstr

    # with open('./data/time_cost.txt', 'w+') as file:
    #     file.write(time_str)

    # todo add dtw exams
    # # KNN exams
    # knn_exams(data, training_sample_ids, test_sample_ids)

    # Incremental exams
    # with open(common_para['path']['data_set_incremental_ids_file'], 'r') as f:
    #     data_set_ids = json.load(f)
    # training_sample_ids = data_set_ids['training_set']
    # test_sample_ids = data_set_ids['test_set']
    # incremental_sample_ids = data_set_ids['incremental_set']
    # incremental_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, incremental_sample_ids, random_seed)

