import os
import os.path as osp
import shutil
from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data

# lzy made
from torch_geometric.data import Data, HeteroData
from typing import List, Optional
from torch import Tensor
def cat(seq: List[Optional[Tensor]]) -> Optional[Tensor]:
    values = [v for v in seq if v is not None]
    values = [v for v in values if v.numel() > 0]
    values = [v.unsqueeze(-1) if v.dim() == 1 else v for v in values]
    return torch.cat(values, dim=-1) if len(values) > 0 else None

def read_FGNN_data(args, ori_dataset):
    FGNN_datalist = {}
    for split_type in ["train", "val", "test"]:
        FGNN_datalist[split_type] = []
        for FGNN_data in ori_dataset[split_type]:
            # if args.pe == "depth":
            #     x = cat([FGNN_data.x, FGNN_data.abs_pe])
            # elif args.pe == "sin_depth":
            #     x = cat([FGNN_data.x, FGNN_data.abs_pe])
            # else:
            x = FGNN_data.x

            if args.struct == "undirected":
                edge_index = torch.cat([FGNN_data.edge_index, FGNN_data.edge_index.flip(0)], dim=1)
            elif args.struct == "forward" or args.struct == "for_back" or args.struct=="for_back_undir":
                edge_index = FGNN_data.edge_index
            else: 
                print("[INFO] in file tu_dataset.py read_FGNN_data function :wrong struct")
                exit()
            data_new = Data(x=x, edge_index=edge_index, edge_attr=torch.empty((edge_index.size(1), 0)),\
                            FA=FGNN_data.FA, type=FGNN_data.type, node_label = FGNN_data.node_label, abs_depth=FGNN_data.abs_pe, RWPE = FGNN_data.RWPE, Eigvecs= FGNN_data.Eigvecs)
            FGNN_datalist[split_type].append(data_new)
    return FGNN_datalist

def homo_2_hyter(dataset):
    new_dataset = []
    for data in tqdm(dataset):
        # hetero_data = HeteroData()
        node_types = torch.argmax(data.x[:, :3], dim=1) 
        edge_index = data.edge_index
        
        # for node_type in torch.unique(node_types):
        #     hetero_data[f'node_type_{node_type.item()}'].x = data.x[node_types == node_type]

        edge_class = [
            (0, 0),
            (0, 1),  # node_type_0 -> node_type_1
            (0, 2),  # node_type_0 -> node_type_2
            (1, 1),
            (1, 0),  # node_type_1 -> node_type_0
            (1, 2),  # node_type_1 -> node_type_2
            (2, 2),
            (2, 0),  # node_type_2 -> node_type_0
            (2, 1),  # node_type_2 -> node_type_1
        ]

        edge_types = torch.empty(edge_index.size(1), dtype=torch.long)
        for i, (src, dst) in enumerate(edge_class):
            src_type = src
            dst_type = dst

            for j, (edge_src, edge_dst) in enumerate(edge_index.T):
                if node_types[edge_src.item()] == src_type and node_types[edge_dst.item()] == dst_type:
                    edge_types[j] = i
        data.edge_type = edge_types
        new_dataset.append(data)
        # for src_type, dst_type in edges:
        #     edge_index = data.edge_index[:, (node_types[data.edge_index[0]] == src_type) & 
        #                                 (node_types[data.edge_index[1]] == dst_type)]
        #     hetero_data[f'node_type_{src_type}', 'to', f'node_type_{dst_type}'].edge_index = edge_index

        # homogeneous_graph = hetero_data.to_homogeneous()
        # new_dataset.append(hetero_data)

    # print(new_dataset)
    return new_dataset

class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    """

    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, args, ori_dataset, root, name, transform=None, pre_transform=None,
                 pre_filter=None, use_node_attr=False, use_edge_attr=False,
                 cleaned=False):
        self.name = name
        self.cleaned = cleaned
        self.ori_dataset = ori_dataset
        self.args = args
        super(TUDataset, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url('{}/{}.zip'.format(url, self.name), folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        # lzy change
        # self.data, self.slices, size, FGNN_datalist = read_tu_data(self.raw_dir, self.name, self.ori_dataset)
        FGNN_datalist = read_FGNN_data(self.args, self.ori_dataset)

        
        # FGNN_dataset = self.ori_dataset

        if self.pre_filter is not None:
            # data_list = [self.get(idx) for idx in range(len(self))]
            data_list = FGNN_datalist["train"] + FGNN_datalist["val"] + FGNN_datalist["test"]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        # if self.pre_transform is not None:
        #     data_list = [self.get(idx) for idx in range(len(self))]
        #     #data_list = [self.pre_transform(data) for data in data_list]
        #     new_data_list = []

        #     # self.pre_transform(FGNN_dataset["train"][0])

        #     for data in tqdm(data_list):
        #         new_data_list.append(self.pre_transform(data))
        #     data_list = new_data_list
        #     self.data, self.slices = self.collate(data_list)
        if self.pre_transform is not None:
            data_list = FGNN_datalist["train"] + FGNN_datalist["val"] + FGNN_datalist["test"]
            # data_list=  data_list[:10]
            data_list = homo_2_hyter(data_list)
            #data_list = [self.pre_transform(data) for data in data_list]
            new_data_list = []

            # self.pre_transform(FGNN_dataset["train"][0])

            for data in tqdm(data_list):
                new_data_list.append(self.pre_transform(data))
            data_list = new_data_list
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
