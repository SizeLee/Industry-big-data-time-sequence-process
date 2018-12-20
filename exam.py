import numpy as np
import random
import LSTM_models

if __name__ == '__main__':
    data = np.load('./data/data_with_label.npz')
    # print(data['features'])
    # print(data['labels'])
    sequence_fix_length = 30
    foresight_steps = 6
    ### todo generate training samples' id
    sample_num = data['labels'].shape[0]
    sample_ids = [i for i in range(sample_num)]
    training_sample_ids = random.sample(sample_ids, int(0.7*sample_num))
    test_sample_ids = list(set(sample_ids) - set(training_sample_ids))
    lstm = LSTM_models.FixedLengthRNN(sequence_fix_length, data['features'].shape[1], class_num=4, cell_type='lstm')
    lstm.train(data['features'], data['labels'], 1, 32, training_sample_ids, True)
    lstm.test(data['features'], data['labels'], test_sample_ids)
    lstm.save_model('./ini_test')



