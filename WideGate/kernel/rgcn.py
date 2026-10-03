import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import RGCNConv, global_mean_pool, global_add_pool, RGCNConv

from torch_geometric.nn import GATConv

import math
from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.conv.dir_gnn_conv import DirGNNConv
import copy as copys
from torch import Tensor

from torch_geometric.data import Data, HeteroData

def homo_2_hyter(dataset):
    new_dataset = []
    for data in dataset:
        hetero_data = HeteroData()
        node_types = torch.argmax(data.x_ori[:, :3], dim=1) 
        edge_index = data.original_edge_index
        
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

class GlobalAttn(torch.nn.Module):
    # def __init__(self, hidden_channels, heads=1, num_layers=2, beta=-1, dropout=0.5, qk_shared=True):
    def __init__(self, hidden_channels, heads=1, num_layers=2, beta=-1, dropout=0.5, qk_shared=True):
        super(GlobalAttn, self).__init__()

        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers
        self.beta = beta
        self.dropout = dropout
        self.qk_shared = qk_shared

        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(num_layers, heads*hidden_channels))
        else:
            self.betas = torch.nn.Parameter(torch.ones(num_layers, heads*hidden_channels)*self.beta)

        self.h_lins = torch.nn.ModuleList()
        if not self.qk_shared:
            self.q_lins = torch.nn.ModuleList()
        self.k_lins = torch.nn.ModuleList()
        self.v_lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        for i in range(num_layers):
            self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            if not self.qk_shared:
                self.q_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.k_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.v_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
        self.lin_out = torch.nn.Linear(heads*hidden_channels, heads*hidden_channels)

    def reset_parameters(self):
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        if not self.qk_shared:
            for q_lin in self.q_lins:
                q_lin.reset_parameters()
        for k_lin in self.k_lins:
            k_lin.reset_parameters()
        for v_lin in self.v_lins:
            v_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)
        self.lin_out.reset_parameters()

    def forward(self, x):
        seq_len, _ = x.size()
        for i in range(self.num_layers):
            h = self.h_lins[i](x)
            k = F.sigmoid(self.k_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
            if self.qk_shared:
                q = k
            else:
                q = F.sigmoid(self.q_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
            v = self.v_lins[i](x).view(seq_len, self.hidden_channels, self.heads)

            # numerator
            kv = torch.einsum('ndh, nmh -> dmh', k, v)
            num = torch.einsum('ndh, dmh -> nmh', q, kv)

            # denominator
            k_sum = torch.einsum('ndh -> dh', k)
            den = torch.einsum('ndh, dh -> nh', q, k_sum).unsqueeze(1)

            # linear global attention based on kernel trick
            if self.beta < 0:
                beta = F.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)
            x = (num/den).reshape(seq_len, -1)
            x = self.lns[i](x) * (h+beta)
            x = F.relu(self.lin_out(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

class DirGNNConv(torch.nn.Module):
    r"""A generic wrapper for computing graph convolution on directed
    graphs as described in the `"Edge Directionality Improves Learning on
    Heterophilic Graphs" <https://arxiv.org/abs/2305.10498>`_ paper.
    :class:`DirGNNConv` will pass messages both from source nodes to target
    nodes and from target nodes to source nodes.

    Args:
        conv (MessagePassing): The underlying
            :class:`~torch_geometric.nn.conv.MessagePassing` layer to use.
        alpha (float, optional): The alpha coefficient used to weight the
            aggregations of in- and out-edges as part of a convex combination.
            (default: :obj:`0.5`)
        root_weight (bool, optional): If set to :obj:`True`, the layer will add
            transformed root node features to the output.
            (default: :obj:`True`)
    """
    def __init__(
        self,
        conv: MessagePassing,
        alpha: float = 0.5,
        root_weight: bool = True,
    ):
        super().__init__()

        self.alpha = alpha
        self.root_weight = root_weight

        self.conv_in = copys.deepcopy(conv)
        self.conv_out = copys.deepcopy(conv)

        if hasattr(conv, 'add_self_loops'):
            self.conv_in.add_self_loops = False
            self.conv_out.add_self_loops = False
        if hasattr(conv, 'root_weight'):
            self.conv_in.root_weight = False
            self.conv_out.root_weight = False

        if root_weight:
            self.lin = torch.nn.Linear(conv.in_channels, conv.out_channels)
        else:
            self.lin = None

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv_in.reset_parameters()
        self.conv_out.reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_types: Tensor) -> Tensor:
        """"""  # noqa: D419
        x_in = self.conv_in(x, edge_index, edge_types)
        x_out = self.conv_out(x, edge_index.flip([0]), edge_types)

        out = self.alpha * x_out + (1 - self.alpha) * x_in

        if self.root_weight:
            out = out + self.lin(x)

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.conv_in}, alpha={self.alpha})'



class NestedDIR_RGCN(torch.nn.Module):
    def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False, num_relations=9, without_srbgraph=True):
        super(NestedDIR_RGCN, self).__init__()
        self.without_subgraph = without_srbgraph
        self.use_rd = use_rd
        self.use_z = use_z
        self.args = args
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset.num_features
        if not without_srbgraph and (self.use_z or self.use_rd):
            input_dim += 8

        # self.conv1 = GINConv(
        #     Sequential(
        #         Linear(input_dim, hidden),
        #         BN(hidden),
        #         ReLU(),
        #         Linear(hidden, hidden),
        #         BN(hidden),
        #         ReLU(),
        #     ),
        #     train_eps=True)

        # 计算num_relations
        # num_relations = torch.unique(dataset.edge_type).shape[0]

        # 增加一个Embedding
        # self.embedding = nn.Linear(in_features=input_dim, out_features= hidden, bias=False)
        self.embedding = nn.Linear(input_dim, out_features= hidden, bias=False)

        # 准备PE
        # MAX_DEPTH是预设的最大层数
        MAX_DEPTH = 510  #经测试，最大有507层的
        self.position = torch.arange(MAX_DEPTH).unsqueeze(1)
        self.div_term = torch.exp(torch.arange(0, hidden, 2) * (-math.log(10000.0) / hidden))
        self.poe = torch.zeros(MAX_DEPTH, hidden)
        self.poe[:, 0::2] = torch.sin(self.position * self.div_term)
        self.poe[:, 1::2] = torch.cos(self.position * self.div_term)


        if args.pe != "None":
            self.conv1 = DirGNNConv(RGCNConv(hidden, hidden, num_relations))
            # self.conv1 = RGCNConv(hidden, hidden, num_relations)
        else:
            self.conv1 = DirGNNConv(RGCNConv(input_dim, hidden, num_relations))
            # self.conv1 = RGCNConv(input_dim, hidden, num_relations)
        self.convs = torch.nn.ModuleList()
        # for i in range(num_layers - 1):
        #     self.convs.append(
        #         GINConv(
        #             Sequential(
        #                 Linear(hidden, hidden),
        #                 BN(hidden), 
        #                 ReLU(),
        #                 Linear(hidden, hidden),
        #                 BN(hidden), 
        #                 ReLU(),
        #             ),
        #             train_eps=True))
        
        for i in range(num_layers - 1):
            self.convs.append(DirGNNConv(RGCNConv(hidden, hidden, num_relations)))
            # self.convs.append(RGCNConv(hidden, hidden, num_relations))


        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

        self.task = args.task
        self.lin_graph_clasify = Linear(hidden, dataset.type.max().item()+1)
        self.lin_graph_regression = Linear(hidden, 1)
        self.lin_node_type = Linear(128, 2)  # ??


        # polynormer
        self.global_attn = GlobalAttn(num_layers * hidden)

    def reset_parameters(self):
        if self.use_rd:
            self.rd_projection.reset_parameters()
        if self.use_z:
            self.z_embedding.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        if not self.without_subgraph:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            edge_types = data.edge_type if hasattr(data, 'edge_type') else None
        else:
            x, edge_index, batch = data.x_ori, data.original_edge_index, data.batch
            # data_new = homo_2_hyter([data])[0]
            edge_types = data.edge_type_ori

        

        # node label embedding
        z_emb = 0
        if self.use_z and 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
        
        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            z_emb += rd_proj

        if not self.without_subgraph and (self.use_rd or self.use_z):
            x = torch.cat([z_emb, x], -1)


        if self.args.pe != "None":
            x = self.embedding(x)

        if self.args.pe == 'depth':
            dagpe = data.abs_depth_
            self.poe = self.poe.to(x.device)
            dagpe = self.poe[:dagpe.shape[0]][dagpe]
            dagpe = dagpe.to(x.device)
            x = x + dagpe
        elif self.args.pe == 'relative_depth':
            dagpe = data.relative_depth_
            self.poe = self.poe.to(x.device)
            dagpe = self.poe[:dagpe.shape[0]][dagpe]
            dagpe = dagpe.to(x.device)
            x = x + dagpe
        elif self.args.pe !="None":
            print("wrong PE")
            exit()

        

        x = self.conv1(x, edge_index, edge_types)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_types)
            xs += [x]

        x = torch.cat(xs, dim=1)
        if not self.without_subgraph:
            x = global_mean_pool(x, data.node_to_subgraph)

        node_embedding = x
        
        # polynormer attn
        # x_local = node_embedding
        # x = self.global_attn(x)

        #x = global_add_pool(x, data.subgraph_to_graph)
        x = global_mean_pool(x, data.subgraph_to_graph)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        graph_embedding = x

        # x = self.lin2(x)

        if self.task == "type":
            return F.log_softmax(self.lin_graph_clasify(graph_embedding), dim=-1)
        elif self.task == "FA":
            return self.lin_graph_regression(graph_embedding)
        elif self.task in ["0", "1", "2", "3", "4", "5", "6"]:
            return F.log_softmax(self.lin_node_type(node_embedding), dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class NestedRGCN(torch.nn.Module):
    def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False, num_relations=9):
        super(NestedRGCN, self).__init__()
        self.use_rd = use_rd
        self.use_z = use_z
        self.args = args
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset.num_features
        if self.use_z or self.use_rd:
            input_dim += 8

        # self.conv1 = GINConv(
        #     Sequential(
        #         Linear(input_dim, hidden),
        #         BN(hidden),
        #         ReLU(),
        #         Linear(hidden, hidden),
        #         BN(hidden),
        #         ReLU(),
        #     ),
        #     train_eps=True)

        # 计算num_relations
        # num_relations = torch.unique(dataset.edge_type).shape[0]

        # 增加一个Embedding
        # self.embedding = nn.Linear(in_features=input_dim, out_features= hidden, bias=False)
        self.embedding = nn.Linear(input_dim, out_features= hidden, bias=False)

        # 准备PE
        # MAX_DEPTH是预设的最大层数
        MAX_DEPTH = 510  #经测试，最大有507层的
        self.position = torch.arange(MAX_DEPTH).unsqueeze(1)
        self.div_term = torch.exp(torch.arange(0, hidden, 2) * (-math.log(10000.0) / hidden))
        self.poe = torch.zeros(MAX_DEPTH, hidden)
        self.poe[:, 0::2] = torch.sin(self.position * self.div_term)
        self.poe[:, 1::2] = torch.cos(self.position * self.div_term)


        if args.pe != "None":
            self.conv1 = RGCNConv(hidden, hidden, num_relations)
        else:
            self.conv1 = RGCNConv(input_dim, hidden, num_relations)
        self.convs = torch.nn.ModuleList()
        # for i in range(num_layers - 1):
        #     self.convs.append(
        #         GINConv(
        #             Sequential(
        #                 Linear(hidden, hidden),
        #                 BN(hidden), 
        #                 ReLU(),
        #                 Linear(hidden, hidden),
        #                 BN(hidden), 
        #                 ReLU(),
        #             ),
        #             train_eps=True))
        
        for i in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden, hidden, num_relations))


        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

        self.task = args.task
        self.lin_graph_clasify = Linear(hidden, dataset.type.max().item()+1)
        self.lin_graph_regression = Linear(hidden, 1)
        self.lin_node_type = Linear(128, 2)  # ??


        # polynormer
        self.global_attn = GlobalAttn(num_layers * hidden)


    def reset_parameters(self):
        if self.use_rd:
            self.rd_projection.reset_parameters()
        if self.use_z:
            self.z_embedding.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_types = data.edge_type if hasattr(data, 'edge_type') else None

        

        # node label embedding
        z_emb = 0
        if self.use_z and 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
        
        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            z_emb += rd_proj

        if self.use_rd or self.use_z:
            x = torch.cat([z_emb, x], -1)


        if self.args.pe != "None":
            x = self.embedding(x)

        if self.args.pe == 'depth':
            dagpe = data.abs_depth_
            self.poe = self.poe.to(x.device)
            dagpe = self.poe[:dagpe.shape[0]][dagpe]
            dagpe = dagpe.to(x.device)
            x = x + dagpe
        elif self.args.pe == 'relative_depth':
            dagpe = data.relative_depth_
            self.poe = self.poe.to(x.device)
            dagpe = self.poe[:dagpe.shape[0]][dagpe]
            dagpe = dagpe.to(x.device)
            x = x + dagpe
        elif self.args.pe !="None":
            print("wrong PE")
            exit()

        

        x = self.conv1(x, edge_index, edge_types)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_types)
            xs += [x]

        x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)

        node_embedding = x
        
        # polynormer attn
        # x_local = node_embedding
        x = self.global_attn(x)

        #x = global_add_pool(x, data.subgraph_to_graph)
        x = global_mean_pool(x, data.subgraph_to_graph)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        graph_embedding = x

        # x = self.lin2(x)

        if self.task == "type":
            return F.log_softmax(self.lin_graph_clasify(graph_embedding), dim=-1)
        elif self.task == "FA":
            return self.lin_graph_regression(graph_embedding)
        elif self.task in ["0", "1", "2", "3", "4", "5", "6"]:
            return F.log_softmax(self.lin_node_type(node_embedding), dim=-1)

    def __repr__(self):
        return self.__class__.__name__
    

class Polynormer(torch.nn.Module):
    def __init__(self, args, dataset, num_layers, hidden, local_layers=3, global_layers=2,
            in_dropout=0.15, dropout=0.5, global_dropout=0.5, heads=1, beta=-1, pre_ln=False, is_global = False):
        super(Polynormer, self).__init__()

        in_channels = dataset[0].x.shape[1]
        hidden_channels = hidden

        self._global = is_global
        self.in_drop = in_dropout
        self.dropout = dropout
        self.pre_ln = pre_ln

        ## Two initialization strategies on beta
        self.beta = beta
        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(local_layers,heads*hidden_channels))
        else:
            self.betas = torch.nn.Parameter(torch.ones(local_layers,heads*hidden_channels)*self.beta)

        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()

        for _ in range(local_layers):
            self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.local_convs.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads,
                concat=True, add_self_loops=False, bias=False))
            self.lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(heads*hidden_channels))

        self.lin_in = torch.nn.Linear(in_channels, heads*hidden_channels)
        self.ln = torch.nn.LayerNorm(heads*hidden_channels)
        self.global_attn = GlobalAttn(hidden_channels, heads, global_layers, beta, global_dropout)
        # self.pred_local = torch.nn.Linear(heads*hidden_channels, out_channels)
        # self.pred_global = torch.nn.Linear(heads*hidden_channels, out_channels)

        self.task = args.task
        self.lin_graph_clasify = Linear(hidden, dataset[0].type.max().item()+1)
        self.lin_graph_regression = Linear(hidden, 1)
        self.lin_node_type = Linear(hidden, 2)  # ??

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.ln.reset_parameters()
        self.global_attn.reset_parameters()
        # self.pred_local.reset_parameters()
        # self.pred_global.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        x = F.dropout(x, p=self.in_drop, training=self.training)
        x = self.lin_in(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        ## equivariant local attention
        x_local = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            h = self.h_lins[i](x)
            h = F.relu(h)
            x = local_conv(x, edge_index) + self.lins[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.beta < 0:
                beta = F.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)
            x = (1-beta)*self.lns[i](h*x) + beta*x
            x_local = x_local + x

        ## equivariant global attention
        if self._global:
            x_global = self.global_attn(self.ln(x_local))
            # x = self.pred_global(x_global)
            x = x_global
        else:
            # x = self.pred_local(x_local)
            x = x_local

        node_embedding = x
        x = global_mean_pool(x, data.batch)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        graph_embedding = x

        if self.task == "type":
            return F.log_softmax(self.lin_graph_clasify(graph_embedding), dim=-1)
        elif self.task == "FA":
            return self.lin_graph_regression(graph_embedding)
        elif self.task in ["0", "1", "2", "3", "4", "5", "6"]:
            return F.log_softmax(self.lin_node_type(node_embedding), dim=-1)

        # return x