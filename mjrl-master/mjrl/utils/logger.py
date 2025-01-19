import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pickle
import os
import csv

class DataLog:

    def __init__(self):
        # 初始化一个空的日志字典
        self.log = {}
        # 初始化最大长度为 0
        self.max_len = 0

    def log_kv(self, key, value):
        # 用于记录键值对

        # 注意：这个实现容易出错，如果在某一次迭代中某些键缺失，会导致不一致
        # 如果键不在日志字典中，创建一个对应键的空列表
        if key not in self.log:
            self.log[key] = []
        # 将值添加到对应键的列表中
        self.log[key].append(value)
        # 如果该键对应的值列表长度大于最大长度，更新最大长度
        if len(self.log[key]) > self.max_len:
            self.max_len = self.max_len + 1

    def save_log(self, save_path):
        # TODO: Validate all lengths are the same.
        pickle.dump(self.log, open(save_path + '/log.pickle', 'wb'))
        with open(save_path + '/log.csv', 'w') as csv_file:
            fieldnames = list(self.log.keys())
            if 'iteration' not in fieldnames:
                fieldnames = ['iteration'] + fieldnames

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in range(self.max_len):
                row_dict = {'iteration': row}
                for key in self.log.keys():
                    if row < len(self.log[key]):
                        row_dict[key] = self.log[key][row]
                writer.writerow(row_dict)

    def get_current_log(self):
        row_dict = {}
        for key in self.log.keys():
            # TODO: this is very error-prone (alignment is not guaranteed)
            row_dict[key] = self.log[key][-1]
        return row_dict

    def shrink_to(self, num_entries):
        for key in self.log.keys():
            self.log[key] = self.log[key][:num_entries]

        self.max_len = num_entries
        assert min([len(series) for series in self.log.values()]) == \
            max([len(series) for series in self.log.values()])

    def read_log(self, log_path):
        assert log_path.endswith('log.csv')

        with open(log_path) as csv_file:
            reader = csv.DictReader(csv_file)
            listr = list(reader)
            keys = reader.fieldnames
            data = {}
            for key in keys:
                data[key] = []
            for row, row_dict in enumerate(listr):
                for key in keys:
                    try:
                        data[key].append(eval(row_dict[key]))
                    except:
                        print("ERROR on reading key {}: {}".format(key, row_dict[key]))

                if 'iteration' in data and data['iteration'][-1] != row:
                    raise RuntimeError("Iteration %d mismatch -- possibly corrupted logfile?" % row)

        self.log = data
        self.max_len = max(len(v) for k, v in self.log.items())
        print("Log read from {}: had {} entries".format(log_path, self.max_len))
