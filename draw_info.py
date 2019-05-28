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

        return

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

        return

    def draw_for_paper(self):
        draw_infos = ['batch_loss', 'batch_accuracy', 'val_accuracy']
        for each in draw_infos:
            self.draw(['lstm', 'gru'], each)

        for each in draw_infos:
            self.draw(['sru', 'cnn', 'cnn_a'], each)

        return

class Draw_incremental:
    def __init__(self):
        self.read_file()
        return

    def read_file(self):
        with open('./data/incremental_result.json', 'r') as f:
            self.info = json.load(f)

    def cal_effective_accuracy(self, iterable_val, iterable_dis, alpha=7):
        v = np.array(iterable_val)
        dis = np.array(iterable_dis)
        effective_v = self._s(v, dis, alpha)
        return effective_v.tolist()

    def _s(self, v, dis, alpha):
        return v * (alpha/(alpha + np.exp(-dis)))

    def draw_for_paper(self, metric_names, thresholds, save_dir='./data/incremental_info/'):
        plot_color = ['blue', 'green', 'red', 'black', 'yellow', 'magenta', 'cyan']
        markers = ['o', '*', '+', 'x', '^', 'p']
        line_widths = [6, 5, 4, 3, 2, 1]
        markers_size = [12, 11, 10, 9, 8, 7]

        for each_metric in metric_names:
            plot_ys = []
            plot_ups = []
            plot_xs = []
            for each_t in thresholds:
                ac = self.info[each_metric][each_t]['ac']
                dis = self.info[each_metric][each_t]['up']
                x = [i+1 for i in range(len(ac))]
                effective_ac = self.cal_effective_accuracy(ac, dis)
                up = [1 if i == 0 else 0 for i in dis]
                plot_xs.append(x)
                plot_ys.append(effective_ac)
                plot_ups.append(up)

            # draw effective accuracy
            plt.figure(figsize=(12, 8), dpi=360)
            lines = []
            for i in range(len(plot_xs)):
                l, = plt.plot(plot_xs[i], plot_ys[i], color=plot_color[i], marker=markers[i])
                # l, = plt.plot(plot_xs[i], plot_ys[i], color=plot_color[i], marker=markers[i], markersize=markers_size[i],
                #               linewidth=line_widths[i])
                lines.append(l)

            plt.legend(handles=lines, labels=thresholds, loc='best')

            plt.xlabel('windows No.')
            plt.ylabel('effective accuracy')
            plt.title('effective accuracy of '+each_metric)

            save_file_name = save_dir + each_metric+'_effective_ac.png'
            plt.savefig(save_file_name, format='png')
            plt.show()

            # draw update state
            plt.figure(figsize=(12, 8), dpi=360)
            lines = []
            for i in range(len(plot_xs)):
                l, = plt.plot(plot_xs[i], plot_ups[i], color=plot_color[i], marker=markers[i], markersize=markers_size[i],
                              linewidth=line_widths[i])
                lines.append(l)

            plt.legend(handles=lines, labels=thresholds, loc='best')

            plt.xlabel('windows No.')
            plt.ylabel('whether update')
            # ax = plt.gca()
            # ax.set_yticklabels(['N', 'Y'])
            plt.yticks([0, 1], ['N', 'Y'])
            plt.title('update record of ' + each_metric)

            save_file_name = save_dir + each_metric + '_update.png'
            plt.savefig(save_file_name, format='png')
            plt.show()

    def find_best_parameter(self, metric_names, thresholds):
        max_effective_ac = 0.
        for each_metric in metric_names:
            # each_metric = 'mean'
            for each_t in thresholds:
                ac = self.info[each_metric][each_t]['ac']
                dis = self.info[each_metric][each_t]['up']
                effective_ac = self.cal_effective_accuracy(ac, dis)
                mean_effective_ac = np.array(effective_ac).mean()
                if mean_effective_ac > max_effective_ac:
                    max_effective_ac = mean_effective_ac
                    best_metrics = each_metric
                    best_threhold = each_t
        print(best_metrics, best_threhold, max_effective_ac)
        print(self.info[best_metrics][best_threhold]['ac'])
        best_ac = np.array(self.info[best_metrics][best_threhold]['ac'])
        print(best_ac.mean(), best_ac.min(), best_ac.max())
        best_up = self.info[best_metrics][best_threhold]['up']
        print(best_up)
        print(len(best_up))
        print(sum((np.array(best_up) == 0)*1))

if __name__ == '__main__':
    painter = Draw_info()
    painter.draw_for_paper()
    painter = Draw_incremental()
    painter.draw_for_paper(['mean', 'min', 'max'], ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    # painter.draw_for_paper(['min'], ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    # painter.draw_for_paper(['mean'], ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    # painter.draw_for_paper(['median'], ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    painter.find_best_parameter(['mean', 'min', 'max', 'median'], ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])


