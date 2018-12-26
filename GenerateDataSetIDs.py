import random
import numpy as np
import json
import configparser

def generate_data_ids(start, end, sequence_len=1, foresight_steps=0):
    real_end = end - sequence_len + 1 - foresight_steps
    sample_num = real_end - start
    sample_ids = [i for i in range(start, real_end)]
    training_sample_ids = random.sample(sample_ids, int(0.7 * sample_num))
    test_sample_ids = list(set(sample_ids) - set(training_sample_ids))
    return training_sample_ids, test_sample_ids

def create_and_save_data_ids():
    cp = configparser.ConfigParser()
    cp.read('common_para.ini')

    data = np.load(cp['path']['data_file'])
    # print(data['features'])
    # print(data['labels'])
    ### generate training samples' id
    sample_num = data['labels'].shape[0]
    training_sample_ids, test_sample_ids = generate_data_ids(0, sample_num,
                                                             cp['common_parameters'].getint('sequence_fix_length'),
                                                             cp['common_parameters'].getint('foresight_steps'))

    # sample_ids = [i for i in range(sample_num)]
    # training_sample_ids = random.sample(sample_ids, int(0.7 * sample_num))
    # test_sample_ids = list(set(sample_ids) - set(training_sample_ids))


    with open(cp['path']['data_set_ids_file'], 'w') as f:
        data_set_ids = {'training_set': training_sample_ids, 'test_set': test_sample_ids}
        json.dump(data_set_ids, f)
    return

if __name__ == '__main__':
    create_and_save_data_ids()