import numpy as np
import BPNN_model
import RNN_models, CNN_models, AttentionModel
import configparser
import json
import time

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
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('lstm_model test start')
    lstm = RNN_models.FixedLengthRNN(sequence_fix_length, data['features'].shape[1],
                                     class_num=class_num, cell_type='lstm')
    # lstm.train(data['features'], data['labels'], 1, 1024, training_sample_ids,
    #            foresight_steps=foresight_steps, reset_flag=True)
    # lstm.test(data['features'], data['labels'], test_sample_ids)
    # lstm.save_model('./model/rnn_model')

    lstm.train_v2(data['features'], data['labels'], data['samples_length'], 1, 1024, training_sample_ids,
               foresight_steps=0, reset_flag=True)
    lstm.test_v2(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    lstm.save_model('./model/lstm_model_v2')

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('lstm_model test over\n')

    # gru
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('gru_model test start')
    gru = RNN_models.FixedLengthRNN(sequence_fix_length, data['features'].shape[1],
                                     class_num=class_num, cell_type='gru')
    # gru.train(data['features'], data['labels'], 1, 1024, training_sample_ids,
    #            foresight_steps=foresight_steps, reset_flag=True)
    # gru.test(data['features'], data['labels'], test_sample_ids)
    # gru.save_model('./model/rnn_model')

    gru.train_v2(data['features'], data['labels'], data['samples_length'], 1, 1024, training_sample_ids,
                  foresight_steps=0, reset_flag=True)
    gru.test_v2(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    gru.save_model('./model/gru_model_v2')

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('gru_model test over\n')

    # sru
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('sru_model test start')
    sru = RNN_models.FixedLengthRNN(sequence_fix_length, data['features'].shape[1],
                                     class_num=class_num, cell_type='sru')
    # sru.train(data['features'], data['labels'], 1, 1024, training_sample_ids,
    #            foresight_steps=foresight_steps, reset_flag=True)
    # sru.test(data['features'], data['labels'], test_sample_ids)
    # sru.save_model('./model/rnn_model')

    sru.train_v2(data['features'], data['labels'], data['samples_length'], 1, 1024, training_sample_ids,
                  foresight_steps=0, reset_flag=True)
    sru.test_v2(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    sru.save_model('./model/sru_model_v2')

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('sru_model test over\n')

    # cnn exams
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('cnn_model test start')
    cnn = CNN_models.ConvSequence2One(sequence_fix_length, data['features'].shape[1],
                                     class_num=class_num)
    # cnn.train(data['features'], data['labels'], 1, 1024, training_sample_ids)
    # cnn.test(data['features'], data['labels'], test_sample_ids)
    # cnn.save_model('./model/cnn_model')

    cnn.train_v2(data['features'], data['labels'], data['samples_length'], 1, 1024, training_sample_ids,
                  foresight_steps=0, reset_flag=True)
    cnn.test_v2(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    cnn.save_model('./model/cnn_model_v2')

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('cnn_model test over\n')

    # attention net exams
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('atn_model test start')
    start_time = time.time()
    atn = AttentionModel.OnlyAttention(sequence_fix_length, data['features'].shape[1], class_num=class_num,
                                       network_hyperparameters='./data/attention_network_hyperparameters_v2.json')

    # atn.train(data['features'], data['labels'], 1, 1024, training_sample_ids)
    # atn.test(data['features'], data['labels'], test_sample_ids)
    # atn.save_model('./model/atn_model_new')

    atn.train_v2(data['features'], data['labels'], data['samples_length'], 2, 1024, training_sample_ids,
                 foresight_steps=0, reset_flag=True)
    atn.test_v2(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    atn.save_model('./model/atn_model_v2')

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    end_time = time.time()
    print('cost time %f' % (end_time - start_time))
    print('atn_model test over\n')


    # traditional ways
    # BPNN exams
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('bpnn_model test start')
    start_time = time.time()
    bpnn = BPNN_model.BPNN(sequence_fix_length, data['features'].shape[1], class_num=class_num)

    # bpnn.train(data['features'], data['labels'], 1, 4096, training_sample_ids,
    #            foresight_steps=foresight_steps, reset_flag=True)
    # bpnn.test(data['features'], data['labels'], test_sample_ids)
    # bpnn.save_model('./model/bpnn_model')

    bpnn.train_v2(data['features'], data['labels'], data['samples_length'], 1, 1024, training_sample_ids,
                  foresight_steps=0, reset_flag=True)
    bpnn.test_v2(data['features'], data['labels'], data['samples_length'], test_sample_ids)
    bpnn.save_model('./model/bpnn_model_v2')

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    end_time = time.time()
    print('cost time %f' % (end_time - start_time))
    print('bpnn_model test over\n')

