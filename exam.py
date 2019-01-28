import numpy as np
import RNN_models, CNN_models
import configparser
import json

if __name__ == '__main__':
    common_para = configparser.ConfigParser()
    common_para.read('common_para.ini')
    sequence_fix_length = common_para['common_parameters'].getint('sequence_fix_length')
    foresight_steps = common_para['common_parameters'].getint('foresight_steps')
    class_num = common_para['common_parameters'].getint('class_num')

    data = np.load(common_para['path']['data_file'])
    # print(data['features'])
    # print(data['labels'])

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
    # lstm = RNN_models.FixedLengthRNN(sequence_fix_length, data['features'].shape[1],
    #                                  class_num=class_num, cell_type='lstm')
    # lstm.train(data['features'], data['labels'], 1, 1024, training_sample_ids,
    #            foresight_steps=foresight_steps, reset_flag=True)
    # lstm.test(data['features'], data['labels'], test_sample_ids)
    # lstm.save_model('./model/ini_test')

    # cnn exams
    cnn = CNN_models.ConvSequence2One(sequence_fix_length, data['features'].shape[1],
                                     class_num=class_num)
    cnn.train(data['features'], data['labels'], 1, 1024, training_sample_ids)
    cnn.test(data['features'], data['labels'], test_sample_ids)


