from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from pathlib import Path
import pandas as pd
from torch_geometric.data import Data, Dataset, InMemoryDataset

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm


def set_seed(args):
    # fix randomseed for reproducing the results
    print('Setting random seed for reproductivity..')
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    cudnn.benchmark = not args.not_cuda_benchmark


def get_parse_args():
    parser = argparse.ArgumentParser(description='Pytorch training script of DeepGate.')
    # dataset settings
    parser.add_argument('--dataset', default='FGNN2', type=str,
                        metavar='NAME', help='target dataset')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--not_cuda_benchmark', action='store_true',
                        help='disable when the input size is not fixed.')
    # parser.add_argument('--data_dir', default='../data/random_circuits',
    #                     type=str, help='the path to the dataset')
    args = parser.parse_args()

    return args


class CountingSimplifyPO(InMemoryDataset):
    def __init__(self, args, transform=None, pre_transform=None, pre_filter=None):
        self.args = args
        self.name = args.dataset
        assert (transform == None) and (pre_transform == None) and (pre_filter == None), "Cannot accept the transform, pre_transfrom and pre_filter args now."
        super(CountingSimplifyPO, self).__init__(args.save_dir, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.args.save_dir

    @property
    def raw_file_names(self):
        return [self.args.circuit_file, self.args.label_file]
        # 返回 raw_dir 目录下的所有文件名
        # return os.listdir(self.root_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass
        # raise NotImplementedError('Download function not implemented.')

    def process(self):
        # 创建一个列表来保存所有的 Data 对象
        data_list = []

        for bench_name in tqdm(os.listdir(str(self.args.root_dir))):
            # full_path = os.path.join(path, entry)
            # # 检查条目是否为目录
            # if os.path.isdir(full_path):
            #     print(entry)
            if self.args.dataset == 'FGNN2':
                label_path = self.args.root_dir.joinpath(bench_name, 'raw_simplify_PO', 'label.csv')
                node_feat_path = self.args.root_dir.joinpath(bench_name, 'raw_simplify_PO', 'node-feat.csv')
                edge_path = self.args.root_dir.joinpath(bench_name, 'raw_simplify_PO', 'signed_edge.csv')
            elif self.args.dataset in ['csa', 'booth']:
                if not bench_name.endswith('_root') or "7nm_mapped" not in bench_name:
                    continue
                label_path = self.args.root_dir.joinpath(bench_name, 'raw_simplify_PO', 'A_out.csv')
                node_feat_path = self.args.root_dir.joinpath(bench_name, 'raw_simplify_PO', 'node-feat.csv')
                edge_path = self.args.root_dir.joinpath(bench_name, 'raw_simplify_PO', 'signed_edge.csv')

            # print(label_path)
            # 读取标签
            

            # 然后读取剩余的行作为节点级标签
            

            # 计算额外的节点标签
            if self.args.dataset in ['csa', 'booth']:
                node_labels = torch.tensor(pd.read_csv(label_path, sep=',', header=None).values,
                                       dtype=torch.long)  # FA_in，FA_out，HA_in，HA_out
            else:
                with open(label_path, 'r') as file:
                    first_line = file.readline().strip().split(',')
                    graph_label_FA = int(first_line[0])  # FA_cnt, HA_cnt
                    graph_label_HA = int(first_line[1])
                    if bench_name.startswith('adder'):
                        graph_label_type = 0
                    elif bench_name.startswith('subtractor'):
                        graph_label_type = 1
                    elif bench_name.startswith('multiplier'):
                        graph_label_type = 2
                    elif bench_name.startswith('divider'):
                        graph_label_type = 3
                    else:
                        graph_label_type = -1

                node_labels = torch.tensor(pd.read_csv(label_path, sep=',', header=None, skiprows=[0]).values,
                                       dtype=torch.long)  # FA_in，FA_out，HA_in，HA_out
                A_in = torch.logical_or(node_labels[:, 0], node_labels[:, 2]).long()
                A_out = torch.logical_or(node_labels[:, 1], node_labels[:, 3]).long()
                in_out = torch.logical_or(A_in, A_out).long()
                node_labels = torch.cat((node_labels, A_in.unsqueeze(-1), A_out.unsqueeze(-1), in_out.unsqueeze(-1)), dim=1)
            # 读取节点特征
            node_features = pd.read_csv(node_feat_path, header=None).values

            # 读取边信息
            edges = pd.read_csv(edge_path, header=None).values
            edge_index = torch.tensor(edges[:, :2].T, dtype=torch.long)  # 转换为 COO 格式
            edge_attr = torch.tensor(edges[:, 2], dtype=torch.long)  # 边属性

            # 创建 Data 对象
            if self.args.dataset in ['csa', 'booth']:
                data = Data(x=torch.tensor(node_features, dtype=torch.float),
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            num_nodes=node_features.shape[0])
            else:
                data = Data(x=torch.tensor(node_features, dtype=torch.float),
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            y=torch.tensor([graph_label_type, graph_label_FA, graph_label_HA], dtype=torch.long),
                            num_nodes=node_features.shape[0])

            # 添加节点标签作为额外属性
            # data.node_labels = torch.tensor(node_labels.flatten(), dtype=torch.long)
            data.node_label = node_labels
            data.name = bench_name

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        if self.pre_filter is None:
            self.data, self.slices = self.collate(data_list)
            torch.save((self.data, self.slices), self.processed_paths[0])

    # def len(self):
    #     return len(self.processed_file_names)

    # def get(self, idx):
    #     data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
    #     return data

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


def main(args):
    if args.dataset == 'FGNN2':
        args.root_dir = Path.home().joinpath('AIGDataset', 'FGNN2', 'global_dataset', 'counting_not_edge')
        args.save_dir = Path.home().joinpath('AIGDataset', 'FGNN2', 'global_dataset', 'counting_not_edge_npz')
        # args.split_file_path = Path.home().joinpath('AIGDataset', 'FGNN2', 'global_dataset', 'counting_bench_split', args.split_file_path)
    elif args.dataset == ['csa', 'booth']:
        args.root_dir = Path.home().joinpath('AIGDataset', 'Gamora', args.dataset)
        args.save_dir = Path.home().joinpath('AIGDataset', 'Gamora', f'{args.dataset}_counting_not_edge_npz')
    os.makedirs(args.save_dir, exist_ok=True)
    args.circuit_file = "graphs.npz"
    args.label_file = "labels.npz"
    print('==> Loading dataset from: ', args.root_dir)
    dataset = CountingSimplifyPO(args)
    print(f'Dataset has {len(dataset)} graphs.')
    print(dataset[0])

def get_FGNN_dataset(args, split_file):
    # args.root_dir = Path.home().joinpath('AIGDataset', 'FGNN2', 'global_dataset', 'counting_not_edge')
    # args.save_dir = Path.home().joinpath('AIGDataset', 'FGNN2', 'global_dataset', 'counting_not_edge_npz')
    args.root_dir = str(Path.home().joinpath('large2', 'large1_save', 'AIGDataset', 'AIGDataset', 'FGNN2', 'global_dataset', 'counting_not_edge'))
    args.save_dir = str(Path.home().joinpath('large2', 'large1_save', 'AIGDataset', 'AIGDataset', 'FGNN2', 'global_dataset', 'counting_not_edge_npz_lzy'))
    os.makedirs(args.save_dir, exist_ok=True)
    args.circuit_file = "graphs.npz"
    args.label_file = "labels.npz"
    args.dataset = "FGNN2"
    print('==> Loading dataset from: ', args.root_dir)
    dataset = CountingSimplifyPO(args)
    print(f'Dataset has {len(dataset)} graphs.')

    split_list = {}
    for type in ["train.txt", "valid.txt", "test.txt"]:
        with open(os.path.join(split_file, type)) as f:
            name_list = []
            lines = f.readlines()
            for line in lines:
                name_list.append(line[:-1])  # remove \n
            split_list[type.split('.')[0]] = name_list

    train_dataset = []
    val_dataset = []
    test_dataset = []
    for data in dataset:
        if data.name in split_list["train"]:
            train_dataset.append(data)
        elif data.name in split_list["valid"]:
            val_dataset.append(data)
        elif data.name in split_list["test"]:
            test_dataset.append(data)
        
    if len(train_dataset) != len(split_list["train"]):
        print(len(train_dataset), len(split_list["train"]))
        print(f"some train circuit doesn't exits in original data")
        exit()
    if len(val_dataset) != len(split_list["valid"]):
        print(f"some val circuit doesn't exits in original data")
        exit()
    if len(test_dataset) != len(split_list["test"]):
        print(f"some test circuit doesn't exits in original data")
        exit()
    # self.train_dataset = train_dataset
    # self.val_dataset = val_dataset
    # self.test_dataset = test_dataset


    return train_dataset, val_dataset, test_dataset

def get_Gamora_dataset(args, dataset):
    # data_map = {
    #     "c6": "c6/mult64_7nm_mapped_root",
    #     "c1": "c1/mult128_7nm_mapped_root",
    #     "c2": "c2/mult258_7nm_mapped_root",
    #     "c5": "c5/mult512_7nm_mapped_root",
    #     "b6": "c6/booth_mult64_7nm_mapped_root",
    #     "b1": "b1/booth_mult128_7nm_mapped_root",
    #     "b2": "b2/booth_mult258_7nm_mapped_root",
    #     "b5": "b5/booth_mult512_7nm_mapped_root"
    # }
    args.dataset = dataset
    # circuit_name = data_map[circuit_name]
    # args.root_dir = Path.home().joinpath("lzy/Gamora_related", circuit_name ,"counting_simplify_PO")
    # args.save_dir = Path.home().joinpath("lzy/Gamora_related", circuit_name ,"counting_simplify_PO_npz")
    
    args.root_dir = Path.home().joinpath("lzy/Gamora_related", args.dataset)
    args.save_dir = Path.home().joinpath("lzy/Gamora_related", f'{args.dataset}_counting_not_edge_npz')
    
    os.makedirs(args.save_dir, exist_ok=True)
    args.circuit_file = "graphs.npz"
    args.label_file = "labels.npz"
    
    print('==> Loading dataset from: ', args.root_dir)
    dataset = CountingSimplifyPO(args)
    print(f'Dataset has {len(dataset)} graphs.')

    return dataset

    # split_list = {}
    # for type in ["train.txt", "valid.txt", "test.txt"]:
    #     with open(os.path.join(split_file, type)) as f:
    #         name_list = []
    #         lines = f.readlines()
    #         for line in lines:
    #             name_list.append(line[:-1])  # remove \n
    #         split_list[type.split('.')[0]] = name_list

    # train_dataset = []
    # val_dataset = []
    # test_dataset = []
    # for data in dataset:
    #     if data.name in split_list["train"]:
    #         train_dataset.append(data)
    #     elif data.name in split_list["valid"]:
    #         val_dataset.append(data)
    #     elif data.name in split_list["test"]:
    #         test_dataset.append(data)
        
    # if len(train_dataset) != len(split_list["train"]):
    #     print(len(train_dataset), len(split_list["train"]))
    #     print(f"some train circuit doesn't exits in original data")
    #     exit()
    # if len(val_dataset) != len(split_list["valid"]):
    #     print(f"some val circuit doesn't exits in original data")
    #     exit()
    # if len(test_dataset) != len(split_list["test"]):
    #     print(f"some test circuit doesn't exits in original data")
    #     exit()
    # # self.train_dataset = train_dataset
    # # self.val_dataset = val_dataset
    # # self.test_dataset = test_dataset


    # return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':

    split_type = ["0.01-0.01-0.98", "0.02-0.02-0.96", "0.05-0.05-0.9", "0.1-0.1-0.8"]
    DATA_DIR = "/home/liujiawei/lzy/DAG_code/dataset/FGNN"
    split_file = "/home/liujiawei/lzy/FGNN_relate/counting_bench_split/" + split_type[0]
    label_dir = "/home/liujiawei/lzy/FGNN_relate/counting_bench_label"

    args = get_parse_args()
    # set_seed(args)
    # main(args)
    # print(get_FGNN_dataset(args, split_file=split_file))
    get_Gamora_dataset(args, "booth")
