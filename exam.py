import numpy as np
import BPNN_model, KNN_sequence_Model
import RNN_models, CNN_models, AttentionModel
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
                   test_sample_ids, foresight_steps=0, reset_flag=True, random_seed=random_seed)
    end_time = time.time()
    print('cost training time %f' % (end_time - start_time))
    model.test_v2(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    model.save_model('./model/%s_model_v2' % cell_type)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # print('cost time %f' % (end_time - start_time))
    print('%s_model test over\n' % cell_type)
    return


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
                 foresight_steps=0, reset_flag=True, record_flag=True, random_seed=random_seed)
    end_time = time.time()
    print('cost training time %f' % (end_time - start_time))
    cnn.test_v2(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    cnn.save_model('./model/cnn_model_v2')

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # print('cost time %f' % (end_time - start_time))
    print('cnn_model test over\n')
    return


def atn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, random_seed=None):
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('atn_model test start')

    atn = AttentionModel.OnlyAttention(sequence_fix_length, data['features'].shape[1], class_num=class_num,
                                       network_hyperparameters='./data/attention_network_hyperparameters_v2.json')

    # atn.train(data['features'], data['labels'], 1, 1024, training_sample_ids)
    # atn.test(data['features'], data['labels'], test_sample_ids)
    # atn.save_model('./model/atn_model_new')
    start_time = time.time()
    atn.train_v2(data['features'], data['labels'], data['samples_length'], 1, 1024, training_sample_ids, test_sample_ids,
                 foresight_steps=0, reset_flag=True, record_flag=True, random_seed=random_seed)
    end_time = time.time()
    print('cost training time %f' % (end_time - start_time))
    atn.test_v2(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    atn.save_model('./model/atn_model_v2')

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # print('cost time %f' % (end_time - start_time))
    print('atn_model test over\n')
    return


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
                  foresight_steps=0, reset_flag=True, record_flag=True, random_seed=random_seed)
    end_time = time.time()
    print('cost training time %f' % (end_time - start_time))
    bpnn.test_v2(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    bpnn.save_model('./model/bpnn_model_v2')

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # print('cost time %f' % (end_time - start_time))
    print('bpnn_model test over\n')
    return

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

    # rnn exams
    # lstm
    rnn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, 'lstm', random_seed)

    # gru
    rnn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, 'gru', random_seed)

    # sru
    rnn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, 'sru', random_seed)

    # cnn exams
    cnn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, random_seed)

    # attention net exams
    atn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, random_seed)

    # traditional ways
    # BPNN exams
    bpnn_exams(sequence_fix_length, foresight_steps, class_num, data, training_sample_ids, test_sample_ids, random_seed)

    # KNN exams
    knn_exams(data, training_sample_ids, test_sample_ids)