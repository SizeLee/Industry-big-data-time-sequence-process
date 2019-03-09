from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import configparser


class  Confusion_matrix_painter:
    def __init__(self):
        return

    def _plot_confusion_matrix(self, cm, labels, title='Confusion Matrix', cmap=plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def plot_and_save_normalized(self, y_predict, y_true, class_num, file_name, save_dir='./data/confusion_matrix'):
        labels = [i for i in range(class_num)]
        tick_marks = np.array(range(len(labels))) + 0.5
        cm = confusion_matrix(y_true, y_predict)
        # np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm_normalized)
        plt.figure(figsize=(12, 8), dpi=120)

        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)

        thresh = cm_normalized.max() / 2.
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm_normalized[y_val][x_val]
            plt.text(x_val, y_val, "%0.6f" % (c,), color="white" if c > thresh else "black",
                         fontsize=7, va='center', ha='center')

        # offset the tick
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)

        self._plot_confusion_matrix(cm_normalized, labels, title=file_name + ' normalized confusion matrix')
        # show confusion matrix
        if save_dir[-1] != '/':
            save_dir = save_dir + '/'
        save_file_name = save_dir + 'normalized ' + file_name
        if save_file_name[-4:] != '.png':
            save_file_name = save_file_name + '.png'
        plt.savefig(save_file_name, format='png')
        plt.show()

        return

    def plot_and_save(self, y_predict, y_true, class_num, file_name, save_dir='./data/confusion_matrix'):
        labels = [i for i in range(class_num)]
        tick_marks = np.array(range(len(labels))) + 0.5
        cm = confusion_matrix(y_true, y_predict)

        print(cm)
        plt.figure(figsize=(12, 8), dpi=120)

        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)

        thresh = cm.max() / 2.
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%0.2d" % (c,), color="white" if c > thresh else "black",
                     fontsize=7, va='center', ha='center')

        # offset the tick
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)

        self._plot_confusion_matrix(cm, labels, title=file_name + ' confusion matrix')
        # show confusion matrix
        if save_dir[-1] != '/':
            save_dir = save_dir + '/'
        save_file_name = save_dir + file_name
        if save_file_name[-4:] != '.png':
            save_file_name = save_file_name + '.png'
        plt.savefig(save_file_name, format='png')
        plt.show()

        return

if __name__ == '__main__':
    common_para = configparser.ConfigParser()
    common_para.read('common_para.ini')
    confusion_matrix_file_path = common_para['path']['confusion_matrix_path']
    class_num = common_para['common_parameters'].getint('class_num')
    dir_name = os.listdir(confusion_matrix_file_path)
    painter = Confusion_matrix_painter()
    for each_dir in dir_name:
        each_complete_dir = confusion_matrix_file_path + each_dir + '/'
        file_name = os.listdir(each_complete_dir)
        for each_file in file_name:
            file = each_complete_dir + each_file
            pre_and_labels = np.load(file)
            # new_y_predict = pre_and_labels['y_predict'].reshape([-1])
            # y_true = pre_and_labels['y_true']
            # np.savez(file, y_predict=new_y_predict, y_true=y_true)
            title_name = each_file[:-12]
            painter.plot_and_save(pre_and_labels['y_predict'], pre_and_labels['y_true'], class_num, title_name,
                                  save_dir=each_complete_dir)
            painter.plot_and_save_normalized(pre_and_labels['y_predict'], pre_and_labels['y_true'], class_num,
                                             title_name, save_dir=each_complete_dir)





