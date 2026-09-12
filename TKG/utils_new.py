#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/8/10 8:15
# @Author : ZM7
# @File : utils_new
# @Software: PyCharm

import os
from torch.utils.data import Dataset, DataLoader

def load_data(data_path):
    data_dir = []
    dir_list = os.listdir(data_path)
    dir_list.sort()
    for filename in dir_list:
        data_dir.append(os.path.join(data_path, filename))
    return data_dir


class myFloder_new(Dataset):
    def __init__(self, root_dir, loader):
        self.root = root_dir
        self.loader = loader
        self.dir_list = load_data(root_dir)
        self.size = len(self.dir_list)

    # def __getitem__(self, index):
    #     dir_ = self.dir_list[index]
    #     try:
    #         data = self.loader(dir_)
    #         if data is None:
    #             raise ValueError(f"Failed to load data from {self.dir_list[index]}")
    #         return data
    #     except Exception as e:
    #         print(f"Error loading file {self.dir_list[index]}: {str(e)}")
    #         raise
    #     # return data
    def __getitem__(self, index):
        max_retries = 3  # 最大重试次数
        retry_count = 0

        while retry_count < max_retries:
            try:
                data = self.loader(self.dir_list[index])
                if data is None:
                    print(f"Warning: Empty data from {self.dir_list[index]}, trying next file")
                    index = (index + 1) % len(self.dir_list)  # 循环到下一个文件
                    retry_count += 1
                    continue
                return data
            except Exception as e:
                print(f"Error loading file {self.dir_list[index]}: {str(e)}, trying next file")
                index = (index + 1) % len(self.dir_list)  # 循环到下一个文件
                retry_count += 1

        # 如果所有重试都失败
        raise RuntimeError(f"Failed to load data after {max_retries} attempts")

    def __len__(self):
        return self.size


def collate_new(data, encoder='rgat', decoder='rgat'):
    data_list = {}
    data = data[0]
    data_list['sub_e_graph'] = data[0][0]
    data_list['sub_d_graph'] = data[0][1]
    data_list['pre_e_nid'] = data[1]['pre_e_nid']
    data_list['pre_d_nid'] = data[1]['pre_d_nid']
    data_list['t'] = data[1]['t']
    data_list['triple'] = data[1]['triple']
    data_list['sample_list'] = data[1]['sample_list']
    data_list['time_list'] = data[1]['time_list']
    data_list['list_length'] = data[1]['list_length']
    data_list['t'] = data[1]['t']
    data_list['sample_unique'] = data[1]['sample_unique']
    data_list['time_unique'] = data[1]['time_unique']
    return data_list


