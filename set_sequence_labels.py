import numpy as np

def set_sequence_labels(file_str):
    data = np.load('./data/data_with_label.npz')
    features = data['features']
    labels = data['labels']
    statistic_dic = {0: 0, 1: 0, 2: 0, 3: 0}
    for i in range(1, labels.shape[0]):
        labels[i, 0] = (labels[(i-1), 0] + labels[i, 0] + 1) % 4
        # print(labels[i, 0])
        statistic_dic[labels[i, 0]] += 1

    statistic_dic[labels[0, 0]] += 1

    np.savez(file_str, features=features, labels=labels)
    print(statistic_dic)
    return

if __name__ == '__main__':
    set_sequence_labels('./data/data_with_label_v2.npz')