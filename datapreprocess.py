import numpy as np
import time
import json
if __name__ == '__main__':
    data_origin_file_name = ['A.csv', 'B.csv']
    data_path = './data/'
    title = []
    title_cn = []
    num_arrays = ()
    time_array = []
    time_flag = True
    for each_file in data_origin_file_name:
        with open(data_path + each_file, 'r') as f:
            # file_process = time.time()
            title1 = f.readline().strip().split(',')
            title2 = f.readline().strip().split(',')
            title1.pop(0)
            title2.pop(0)
            title.extend(title1)
            title_cn.extend(title2)
            # print(title1)
            # print(title2)
            temparray = []
            for line in f:
                if len(line) <= 2:
                    continue
                data_on_each_time = line.strip().strip(',').split(',')
                # print(data_on_each_time)
                if time_flag:
                    time_array.append(data_on_each_time[0])
                try:
                    data_on_each_time = list(map(float, data_on_each_time[1:]))
                except:
                    print(data_on_each_time)
                temparray.append(data_on_each_time[1:])
            # file_process = time.time() - file_process
            # print(file_process)

        # ifcopy_s = time.time()
        temparray = np.array(temparray)
        time_flag = False
        # ifcopy = time.time() - ifcopy_s
        # print(ifcopy)
        num_arrays = (*num_arrays, temparray[:, 1:])
        # ifcopy = time.time() - ifcopy_s
        # print(ifcopy)

    try:
        # s = time.time()
        num_array = np.hstack(num_arrays)
        # e = time.time()
        # print(num_array)
        # print(e-s)
        # print(time_array)
        np.savez(data_path + 'features.npz', num_array)
        with open('./data/time.json', 'w+') as f:
            json.dump(time_array, f)

    except:
        print('error!')
        print(num_arrays[0].shape, num_arrays[1].shape)







