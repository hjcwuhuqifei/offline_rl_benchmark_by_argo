
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def smooth(csv_path,weight=0.85):
    data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    scalar = data['Value'].values
    last = -60
    index = 0
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
        index += 1
        if index >= 100:
            break

    return data['Step'].values, smoothed

def smooth_(csv_path,weight=0.85):
    data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    scalar = data['Value'].values
    last = -60
    index = 0
    smoothed = []
    if len(scalar) > 100:
        i = 0
        while i < len(scalar):
            smoothed_val = last * weight + (1 - weight) * scalar[i]
            smoothed.append(smoothed_val)
            last = smoothed_val
            index += 1
            i += 2
    else:
        for point in scalar:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
            index += 1
        # if index >= 100:
        #     break

    return data['Step'].values, smoothed


if __name__=='__main__':
    root_path = 'data_train_by_td3'
    dir_list = os.listdir(root_path)
    data_list = {}
    step_normal = np.array([])
    for path in dir_list:
        each_alg_path = os.path.join(root_path, path)
        csv_list = os.listdir(each_alg_path)
        data = []
        step = np.array([])
        if path == 'IQL' or path == 'CQL':
            for csv_path in csv_list:
                csv_step, csv_data = smooth_(os.path.join(each_alg_path, csv_path))
                data = data + csv_data
        else:
            for csv_path in csv_list:
                csv_step, csv_data = smooth(os.path.join(each_alg_path, csv_path))
                data = data + csv_data
                if len(csv_step) > 100:
                    csv_step = csv_step[:-1]
                step = np.concatenate((step, csv_step))
        if path == 'AWAC':
            step_normal = step
        data_list.update({path + '_frame':step_normal, path + '_reward':data})

    save = pd.DataFrame(data_list)
    save.to_csv(os.path.join(root_path, 'smooth.csv'))





