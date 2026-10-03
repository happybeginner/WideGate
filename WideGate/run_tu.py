import os.path as osp
import os, sys
import time
from shutil import copy, rmtree
from itertools import product
import pdb
import argparse
import random
import torch
import numpy as np
from kernel.datasets import get_dataset
from kernel.train_eval import cross_validation_with_val_set
from kernel.train_eval import cross_validation_without_val_set, cross_validation_only_with_test
# from kernel.gcn import *
# from kernel.rgcn import *
# from kernel.graph_sage import *
# from kernel.gin import *
# from kernel.gat import *
from kernel.graclus import Graclus
from kernel.top_k import TopK
from kernel.diff_pool import *
from kernel.global_attention import GlobalAttentionNet
from kernel.set2set import Set2SetNet
from kernel.sort_pool import SortPool

# from FGNN_data.read_less import *
# from kernel.other_design import *
from kernel.designed_conv import SAGE_MULT, NestedDIR_Designed,NestedDIR_GIN_Designed, NestedDIR_GAT_Designed, NestedDIR_SAGE_Designed, NestedDIR_RGCN_Designed, homo_2_hyter_TWO


# used to traceback which code cause warnings, can delete
import traceback
import warnings
import sys

from load_FGNN_data import get_FGNN_dataset, get_Gamora_dataset

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

# General settings.F
parser = argparse.ArgumentParser(description='Nested GNN for TU graphs')
parser.add_argument('--data', type=str, default='MUTAG')
parser.add_argument('--clean', action='store_true', default=False,
                    help='use a cleaned version of dataset by removing isomorphism')
parser.add_argument('--no_val', action='store_true', default=False,
                    help='if True, do not use validation set, but directly report best\
                    test performance.')

# GNN settings.
parser.add_argument('--model', type=str, default='NestedDIR_Designed',   #SAGE_MULT, NestedFA， NestedDIR_RGCN_Designed， NestedDIR_Designed--大规模实验模型 NestedDIR_RGCN 
                    help='GCN, GraphSAGE, GIN, GAT, NestedGCN, NestedGraphSAGE, \
                    NestedGIN, and NestedGAT, and NestedRGCN, Polynormer, \
                        NestedDIR_GCN, NestedDIR_GraphSAGE, NestedDIR_GAT, NestedFA, NestedDIR_RGCN')
parser.add_argument('--layers', type=int, default=8)    # 4
parser.add_argument('--hiddens', type=int, default=32)
# parser.add_argument('--h', type=int, default=None, help='the height of rooted subgraph \
#                     for NGNN models')
parser.add_argument('--h', type=int, default=3, help='the height of rooted subgraph \
                    for NGNN models')
parser.add_argument('--use_rd', action='store_true', default=False, 
                    help='use resistance distance as additional node labels')
# parser.add_argument('--use_rp', type=int, default=1, 
#                     help='use RW return probability as additional node features,\
#                     specify num of RW steps here')
parser.add_argument('--use_rp', type=int, default=0, 
                    help='use RW return probability as additional node features,\
                    specify num of RW steps here')
parser.add_argument('--max_nodes_per_hop', type=int, default=None)


# lzy PE
parser.add_argument('--pe', type=str, default="None")   # relative_depth
# parser.add_argument('--pe', type=str, default="relative_depth") # None
# lzy task
parser.add_argument('--task', type=str, default="5")

parser.add_argument('--device', type=str, default="cuda:0")

parser.add_argument('--split_type', type=int, default=2)

parser.add_argument('--struct', type=str, default="for_back_no_test")   # undirected forward for_back undirected  # 构图那里地参数需要手动改

parser.add_argument('--node_label', type=str, default="spd", 
                    help='apply distance encoding to nodes within each subgraph, use node\
                    labels as additional node features; support "hop", "drnl", "spd", \
                    "spd5", etc. Default "spd"=="spd2".')

parser.add_argument('--task_num', type=int, default=-1)   # relative_depth

parser.add_argument('--lr', type=float, default=1e-2)  # 1e-2

parser.add_argument('--polynormer_is_global', type=bool, default=False)   # relative_depth

parser.add_argument('--test', type=bool, default=False)    # only test model on Gamora

# Training settings.
parser.add_argument('--epochs', type=int, default=200)
# parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)

# Other settings.
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--search', action='store_true', default=False, 
                    help='search hyperparameters (layers, hiddens)')
parser.add_argument('--save_appendix', default='', 
                    help='what to append to save-names when saving results')
# parser.add_argument('--keep_old', action='store_true', default=False,
#                     help='if True, do not overwrite old .py files in the result folder')
parser.add_argument('--keep_old', action='store_true', default=True,
                    help='if True, do not overwrite old .py files in the result folder')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data')
parser.add_argument('--cpu', action='store_true', default=False, help='use cpu')
args = parser.parse_args()

task = [["no", "None"], ["spd", "None"], ["hop", "None"],\
        ["no", "depth"], ["spd", "depth"], ["hop", "depth"],\
        ["no", "relative_depth"], ["spd", "relative_depth"], ["hop", "relative_depth"]]

task_num = args.task_num
if task_num > 0:
    args.node_label, args.pe = task[task_num]
    print(f"----\n{task[task_num]}\n-----\n")

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

file_dir = os.path.dirname(os.path.realpath('__file__'))
if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
args.res_dir = os.path.join(file_dir, 'results/TU{}'.format(args.save_appendix))
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 
if not args.keep_old:
    # backup current main.py, model.py files
    copy('run_tu.py', args.res_dir)
    copy('utils.py', args.res_dir)
    copy('kernel/train_eval.py', args.res_dir)
    copy('kernel/datasets.py', args.res_dir)
    copy('kernel/gcn.py', args.res_dir)
    copy('kernel/gat.py', args.res_dir)
    copy('kernel/graph_sage.py', args.res_dir)
    copy('kernel/gin.py', args.res_dir)
# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

if args.data == 'all':
    datasets = [ 'DD', 'MUTAG', 'PROTEINS', 'PTC_MR', 'ENZYMES']
else:
    datasets = [args.data]

if args.search:
    if args.h is None:
        layers = [2, 3, 4, 5]
        hiddens = [32]
        hs = [None]
    else:
        layers = [3, 4, 5, 6]
        hiddens = [32, 32, 32, 32]
        hs = [2, 3, 4, 5]
else:
    layers = [args.layers]
    hiddens = [args.hiddens]
    hs = [args.h]

if args.model == 'all':
    #nets = [GCN, GraphSAGE, GIN, GAT]
    nets = [NestedGCN, NestedGraphSAGE, NestedGIN, NestedGAT]
else:
    nets = [eval(args.model)]

def logger(info):
    f = open(os.path.join(args.res_dir, 'log.txt'), 'a')
    print(info, file=f)

device = torch.device(
    args.device if torch.cuda.is_available() and not args.cpu else 'cpu'
)

if args.no_val:
    cross_val_method = cross_validation_without_val_set
else:
    cross_val_method = cross_validation_with_val_set

if args.test:
    cross_val_method = cross_validation_only_with_test

# lzy read FGNN

split_type = ["0.01-0.01-0.98", "0.02-0.02-0.96", "0.05-0.05-0.9", "0.1-0.1-0.8"]
# DATA_DIR = "/home/liujiawei/lzy/DAG_code/dataset/FGNN"
split_file = "/home/liujiawei/large2/lzy/Wide_gate_portable/FGNN_relate/counting_bench_split/" + split_type[args.split_type]
# label_dir = "/home/liujiawei/lzy/FGNN_relate/counting_bench_label"
# train_dataset, val_dataset, test_dataset = read_dataset_dag(DATA_DIR, split_file, label_dir)  # 原版数据
train_dataset, val_dataset, test_dataset = get_FGNN_dataset(args, split_file)
# Gamora_dataset = get_Gamora_dataset(args, "booth")  # booth
# choise_cir = 3
# print(Gamora_dataset[choise_cir].name)
# Gamora_dataset = [Gamora_dataset[choise_cir], Gamora_dataset[choise_cir]]


# if args.model == "NestedDIR_RGCN_Designed":
#     train_dataset = homo_2_hyter_TWO(train_dataset)
#     val_dataset = homo_2_hyter_TWO(val_dataset)
#     test_dataset = homo_2_hyter_TWO(test_dataset)

results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)
    log = '-----\n{} - {}'.format(dataset_name, Net.__name__)
    print(log)
    logger(log)
    if args.h is not None:
        combinations = zip(layers, hiddens, hs)
    else:
        combinations = product(layers, hiddens, hs)
    for num_layers, hidden, h in combinations:
        if dataset_name == 'DD' and Net.__name__ == 'NestedGAT' and h >= 5:
            print('NestedGAT on DD will OOM for h >= 5. Skipped.')
            continue
        log = "Using {} layers, {} hidden units, h = {}".format(num_layers, hidden, h)
        print(log)
        logger(log)

        if args.model != 'Polynormer':
        #     dataset = get_dataset(
        #     args,
        #     {"train": train_dataset, "val": val_dataset, "test": test_dataset},
        #     dataset_name, 
        #     Net != DiffPool, 
        #     h, 
        #     args.node_label, 
        #     args.use_rd, 
        #     args.use_rp, 
        #     args.reprocess, 
        #     args.clean, 
        #     args.max_nodes_per_hop,    #尝试削减数据处理流程
        # ) 
            # dataset = train_dataset + val_dataset + test_dataset
            dataset = {}
            dataset["train"] = train_dataset
            dataset["val"] = val_dataset
            dataset["test"] = test_dataset
            if args.test:
                dataset = Gamora_dataset
                args.batch_size = 2
                args.task = "0"
            model = Net(args, dataset["train"], num_layers, hidden, args.node_label!='no', args.use_rd)
        else:
            dataset = train_dataset + val_dataset + test_dataset
            model = Net(args, dataset, num_layers, hidden, is_global=args.polynormer_is_global)
        # model = Net(args, dataset, num_layers, hidden, False, args.use_rd)   #不使用z
        loss, acc, std = cross_val_method(
            args,
            dataset,
            model,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            device=device, 
            logger=logger)
        if loss < best_result[0]:
            best_result = (loss, acc, std)
            best_hyper = (num_layers, hidden, h)

    desc = '{:.3f} ± {:.3f}'.format(
        best_result[1], best_result[2]
    )
    log = 'Best result - {}, with {} layers and {} hidden units and h = {}'.format(
        desc, best_hyper[0], best_hyper[1], best_hyper[2]
    )
    print(log)
    logger(log)
    results += ['{} - {}: {}'.format(dataset_name, model.__class__.__name__, desc)]

log = '-----\n{}'.format('\n'.join(results))
print(cmd_input[:-1])
print(log)
logger(log)

print("----+++\n", args.task, args.struct, args.task_num)
