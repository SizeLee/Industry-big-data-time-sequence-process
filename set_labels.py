import numpy as np
# 发现考量所有维度时，数据分布按照长期时间变化，大致分为三部分，前410621中有410546个为一类，
# 接下来245810中有175981为一类，最后392139中有322297为一类。说明是整体数据随时间慢慢变化。
# 是否可以将每个阶段夹杂进的他类记录视为异常呢？需要了解这部分异常的分布位置：
# 异常位置连续，在第一类末尾和第二类开头夹杂了一些第三类。
# 第三类开头夹杂着一些未知类，可能是第二，也可能是第一。
# 把这部分数据作为异常，应该可以增加分类难度。不知道rnn能否hold住。
# 目前看来，先以分三类为基础，添加3类正常标签，将夹杂于三类中的异类作为异常类。
# 需要统计一类的分布范围，二类的分布范围，三类的分布范围，确定异常类数据。
class ClusterSetLabel:
    def __init__(self, features, class_num = 2, file_str = './data/data_with_label.npz'):
        self.features = features
        self.class_num = class_num
        self.file_str = file_str
        self.label = None
        self.sample_num = self.features.shape[0]
        return

    def clustering(self, clustering_way='kmeans', max_rounds=10):
        k_clusters = []
        judge = None
        if clustering_way == 'kmeans':
            k_clusters, judge = self._k_means(max_rounds)

        count = 0
        for cluster in k_clusters:
            clcount = np.sum(cluster[1]*1)
            print(clcount)
            # print(cluster[0])
            print(cluster[1])
            print(np.sum(cluster[1][count:count+clcount]*1))
            for i in range(count, count+clcount):
                if not cluster[1][i]:
                    print(i, end=' ')
            print()
            print()
            count += clcount

        print(judge)
        return

    def _k_means(self, max_rounds):
        self.label = np.zeros((self.sample_num, 1))
        init_scale = self.sample_num//self.class_num
        k_clusters = []
        judge = None
        for i in range(self.class_num):
            self.label[i*init_scale:(i+1)*init_scale, :] = i
            cluster_index = [i for i in range(i*init_scale, (i+1)*init_scale)]
            cluster_center = np.mean(features[cluster_index, :], axis=0)
            cluster = [cluster_center, cluster_index]
            k_clusters.append(cluster)

        for k_m_iter in range(max_rounds):
            print('round:', k_m_iter)
            little_change_flag = True
            distance = []
            for i in range(self.class_num):
                k_clusters[i][1] = []
                distance.append(np.linalg.norm(self.features - k_clusters[i][0], ord=2, axis=1))
            # print(distance)
            belong2 = np.argmin(distance, axis=0)
            # print(belong2)
            for i in range(self.class_num):
                new_index = belong2 == i
                new_center = np.mean(self.features[new_index, :], axis=0)
                # print(k_clusters[i][0])
                # print(new_center)
                # print(np.linalg.norm(new_center-k_clusters[i][0]))
                if np.linalg.norm(new_center-k_clusters[i][0]) > 1e-3:
                    little_change_flag = False
                k_clusters[i][0] = new_center
                k_clusters[i][1] = new_index

            if little_change_flag:
                break

            incluster_distance = []
            for i in range(self.class_num):
                # incluster_distance.append(np.mean(distance[i][k_clusters[i][1]]))
                incluster_distance.append(np.percentile(distance[i][k_clusters[i][1]], 90))
            incluster_distance = np.array(incluster_distance)

            btclusters_distance = [[np.inf for i in range(self.class_num)] for j in range(self.class_num)]
            for i in range(self.class_num-1):
                for j in range(i+1, self.class_num):
                    temp = np.linalg.norm(k_clusters[i][0] - k_clusters[j][0])
                    btclusters_distance[i][j] = temp
                    btclusters_distance[j][i] = temp
            btclusters_distance = np.array(btclusters_distance)

            mini_bt_distance = np.min(btclusters_distance, axis=0)
            judge = incluster_distance/mini_bt_distance

        return k_clusters, judge

if __name__ == '__main__':
    features = np.load('./data/features.npz')['arr_0']
    # print(features)
    # print(features.shape)
    # features = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    classes_num = 3
    clu = ClusterSetLabel(features, classes_num)
    clu.clustering(max_rounds=100)


