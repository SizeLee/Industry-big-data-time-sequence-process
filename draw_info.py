import matplotlib.pyplot as plt
import numpy as np
import os
import configparser
import json

class Draw_info:
    def __init__(self):
        self.read_file()
        self.whole_step = 5000
        return

    def read_file(self):
        self.info = {}
        with open('./data/info/lstm_info.json', 'r') as f:
            self.info['lstm'] = json.load(f)

        with open('./data/info/gru_info.json', 'r') as f:
            self.info['gru'] = json.load(f)

        with open('./data/info/sru_info.json', 'r') as f:
            self.info['sru'] = json.load(f)

        with open('./data/info/cnn_info.json', 'r') as f:
            self.info['cnn'] = json.load(f)

        with open('./data/info/cnn_a_info.json', 'r') as f:
            self.info['cnn_a'] = json.load(f)

        with open('./data/info/bpnn_info.json', 'r') as f:
            self.info['bpnn'] = json.load(f)

    def draw(self, model_names, draw_info_name, save_dir='./data/info/'):  # draw info name: batch_loss, batch_accuracy, val_accuracy
        plot_color = ['blue', 'green', 'red', 'black', 'yellow', 'cyan']
        plot_ys = []
        plot_xs = []
        for each_model in model_names:
            y = self.info[each_model][draw_info_name]
            x = np.linspace(0, self.whole_step, num=len(y))
            plot_ys.append(y)
            plot_xs.append(x)

        plt.figure(figsize=(12, 8), dpi=360)
        lines = []
        for i in range(len(model_names)):
            l, = plt.plot(plot_xs[i], plot_ys[i], color=plot_color[i])
            # print(l)
            lines.append(l)

        # if draw_info_name == 'batch_loss':
        #     location = 'upper right'
        # else:
        #     location = 'lower right'
        # plt.legend(handles=lines, labels=model_names, loc=location)
        plt.legend(handles=lines, labels=model_names, loc='best')

        plt.xlabel('training step')
        plt.ylabel(draw_info_name)
        plt.title(draw_info_name+' change when training')

        model_str = ''
        for each in model_names:
            model_str += each
        file_name = save_dir+model_str+'_'+draw_info_name+'.png'

        plt.savefig(file_name, format='png')
        plt.show()

    def draw_for_paper(self):
        draw_infos = ['batch_loss', 'batch_accuracy', 'val_accuracy']
        for each in draw_infos:
            self.draw(['lstm', 'gru'], each)

        for each in draw_infos:
            self.draw(['sru', 'cnn', 'cnn_a'], each)

if __name__ == '__main__':
    painter = Draw_info()
    painter.draw_for_paper()

