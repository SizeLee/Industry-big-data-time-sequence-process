import numpy as np

if __name__ == '__main__':
    data = np.load('./data/data_with_label.npz')
    class_distribution = {}
    labels = data['labels']
    for i in range(labels.shape[0]):
        class_distribution[labels[i, 0]] = class_distribution.get(labels[i, 0], 0) + 1
    print(class_distribution)