import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool, RGCNConv
from torch_geometric.nn.conv.dir_gnn_conv import DirGNNConv
from torch_geometric.nn.conv.fa_conv import FAConv
from torch import nn
import math

import copy as copys
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor


class DirGNNConv_for_FA(torch.nn.Module):
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
            self.lin = torch.nn.Linear(conv.channels, conv.channels)
        else:
            self.lin = None

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv_in.reset_parameters()
        self.conv_out.reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Tensor, x_0: Tensor,  edge_index: Tensor) -> Tensor:
        """"""  # noqa: D419
        x_in = self.conv_in(x, x_0, edge_index)
        x_out = self.conv_out(x, x_0, edge_index.flip([0]))

        out = self.alpha * x_out + (1 - self.alpha) * x_in

        if self.root_weight:
            out = out + self.lin(x)

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.conv_in}, alpha={self.alpha})'

class NestedDIR_FA(torch.nn.Module):
    def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False, num_relations=9):
        super(NestedDIR_FA, self).__init__()
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


        self.lin_pre = Linear(input_dim, hidden)
        if args.pe != "None":
            self.conv1 = DirGNNConv_for_FA(FAConv(hidden))
        else:
            self.conv1 = DirGNNConv_for_FA(FAConv(hidden))
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
            self.convs.append(DirGNNConv_for_FA(FAConv(hidden)))


        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

        self.task = args.task
        self.lin_graph_clasify = Linear(hidden, dataset.type.max().item()+1)
        self.lin_graph_regression = Linear(hidden, 1)
        self.lin_node_type = Linear(128, 2)  # ??


        # polynormer
        # self.global_attn = GlobalAttn(num_layers * hidden)

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
        # 1, 2.
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # 3.
        x, edge_index, batch = data.x_ori, data.original_edge_index, data.batch
        
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
            root_node_mask = data.z[:, 0] == 1
        # x = x[root_node_mask]
            z_emb = z_emb[root_node_mask]
            x = torch.cat([z_emb, x], -1)

        x = self.lin_pre(x)
        x_0 = x
        x = F.relu(self.conv1(x, x_0, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, x_0, edge_index))
            xs += [x]

        x = torch.cat(xs, dim=1)
        # 1.
        # root_node_mask = data.z[:, 0] == 1
        # x = x[root_node_mask]
        #2.
        # x = global_mean_pool(x, data.node_to_subgraph)
        #3. 空

        node_embedding = x
        #x = global_add_pool(x, data.subgraph_to_graph)
        x = global_mean_pool(x, data.subgraph_to_graph)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        graph_embedding = x

        if self.task == "type":
            return F.log_softmax(self.lin_graph_clasify(graph_embedding), dim=-1)
        elif self.task == "FA":
            return self.lin_graph_regression(graph_embedding)
        elif self.task in ["0", "1", "2", "3", "4", "5", "6"]:
            return F.log_softmax(self.lin_node_type(node_embedding), dim=-1)

    def __repr__(self):
        return self.__class__.__name__
    
    def save_model(self, save_path):
        """
        保存模型参数
        """
        torch.save(self.state_dict(), save_path)

    def load_model(self, load_path):
        """
        加载模型参数
        """
        self.load_state_dict(torch.load(load_path))



class NestedFA(torch.nn.Module):
    def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False, without_subgraph=True, undir = False):
        super(NestedFA, self).__init__()
        self.undir = undir
        self.without_subgraph = without_subgraph

        self.use_rd = use_rd
        self.use_z = use_z
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset[0].num_features
        if not without_subgraph and (self.use_z or self.use_rd):
            input_dim += 8

        self.lin_pre = Linear(input_dim, hidden)
        self.conv1 = FAConv(hidden)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(FAConv(hidden))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        # self.lin2 = Linear(hidden, dataset[0].num_classes)

        self.task = args.task
        self.lin_graph_clasify = Linear(hidden, 4)
        self.lin_graph_regression = Linear(hidden, 1)
        self.lin_node_type = Linear(hidden * num_layers, 2)  # ??

    def reset_parameters(self):
        if self.use_rd:
            self.rd_projection.reset_parameters()
        if self.use_z:
            self.z_embedding.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        # self.lin2.reset_parameters()

    def forward(self, data):
        # 1, 2.
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # 3.

        x, edge_index, batch = data.x_ori, data.original_edge_index, data.batch
        if self.undir:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # node label embedding
        z_emb = 0
        if self.use_z and 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
        
        if not self.without_subgraph and (self.use_rd and 'rd' in data):
            rd_proj = self.rd_projection(data.rd)
            z_emb += rd_proj

        if not self.without_subgraph and (self.use_rd or self.use_z):
            root_node_mask = data.z[:, 0] == 1
        # x = x[root_node_mask]
            z_emb = z_emb[root_node_mask]
            x = torch.cat([z_emb, x], -1)

        x = self.lin_pre(x)
        x_0 = x
        x = F.relu(self.conv1(x, x_0, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, x_0, edge_index))
            xs += [x]

        x = torch.cat(xs, dim=1)
        # 1.
        # root_node_mask = data.z[:, 0] == 1
        # x = x[root_node_mask]
        #2.
        if not self.without_subgraph:
            x = global_mean_pool(x, data.node_to_subgraph)
        #3. 空

        node_embedding = x
        #x = global_add_pool(x, data.subgraph_to_graph)
        x = global_mean_pool(x, data.subgraph_to_graph)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        graph_embedding = x

        if self.task == "type":
            return F.log_softmax(self.lin_graph_clasify(graph_embedding), dim=-1)
        elif self.task == "FA":
            return self.lin_graph_regression(graph_embedding)
        elif self.task in ["0", "1", "2", "3", "4", "5", "6"]:
            return F.log_softmax(self.lin_node_type(node_embedding), dim=-1)

    def __repr__(self):
        return self.__class__.__name__
    
    def save_model(self, save_path):
        """
        保存模型参数
        """
        torch.save(self.state_dict(), save_path)

    def load_model(self, load_path):
        """
        加载模型参数
        """
        self.load_state_dict(torch.load(load_path))
    
class SelfGCN(torch.nn.Module):
    def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False):
        super(SelfGCN, self).__init__()
        self.use_rd = use_rd
        self.use_z = use_z
        self.use_z = False    # 不要结构编码了
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset.num_features
        if self.use_z or self.use_rd:
            input_dim += 8

        # self.lin_pre = Linear(input_dim, hidden)
        self.conv1 = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                BN(hidden),
                ReLU(),
                Linear(hidden, hidden),
                BN(hidden),
                ReLU(),
            ),
            train_eps=False)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        BN(hidden), 
                        ReLU(),
                        Linear(hidden, hidden),
                        BN(hidden), 
                        ReLU(),
                    ),
                    train_eps=False))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

        self.task = args.task
        self.lin_graph_clasify = Linear(hidden, dataset.type.max().item()+1)
        self.lin_graph_regression = Linear(hidden, 1)
        self.lin_node_type = Linear(128, 2)  # ??

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
        # 1, 2.
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # 3.
        x, edge_index, batch = data.x_ori, data.original_edge_index, data.batch
        
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
            root_node_mask = data.z[:, 0] == 1
        # x = x[root_node_mask]
            z_emb = z_emb[root_node_mask]
            x = torch.cat([z_emb, x], -1)

        # x = self.lin_pre(x)
        # x_0 = x
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]

        x = torch.cat(xs, dim=1)
        # 1.
        # root_node_mask = data.z[:, 0] == 1
        # x = x[root_node_mask]
        #2.
        # x = global_mean_pool(x, data.node_to_subgraph)
        #3. 空

        node_embedding = x
        #x = global_add_pool(x, data.subgraph_to_graph)
        x = global_mean_pool(x, data.subgraph_to_graph)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        graph_embedding = x

        if self.task == "type":
            return F.log_softmax(self.lin_graph_clasify(graph_embedding), dim=-1)
        elif self.task == "FA":
            return self.lin_graph_regression(graph_embedding)
        elif self.task in ["0", "1", "2", "3", "4", "5", "6"]:
            return F.log_softmax(self.lin_node_type(node_embedding), dim=-1)

    def __repr__(self):
        return self.__class__.__name__
    
class SelfGIN(torch.nn.Module):
    def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False):
        super(SelfGIN, self).__init__()
        self.use_rd = use_rd
        self.use_z = use_z
        self.use_z = False    # 不要结构编码了
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset.num_features
        if self.use_z or self.use_rd:
            input_dim += 8

        # self.lin_pre = Linear(input_dim, hidden)
        self.conv1 = GCNConv(input_dim, hidden)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

        self.task = args.task
        self.lin_graph_clasify = Linear(hidden, dataset.type.max().item()+1)
        self.lin_graph_regression = Linear(hidden, 1)
        self.lin_node_type = Linear(128, 2)  # ??

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
        # 1, 2.
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # 3.
        x, edge_index, batch = data.x_ori, data.original_edge_index, data.batch
        
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
            root_node_mask = data.z[:, 0] == 1
        # x = x[root_node_mask]
            z_emb = z_emb[root_node_mask]
            x = torch.cat([z_emb, x], -1)

        # x = self.lin_pre(x)
        # x_0 = x
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]

        x = torch.cat(xs, dim=1)
        # 1.
        # root_node_mask = data.z[:, 0] == 1
        # x = x[root_node_mask]
        #2.
        # x = global_mean_pool(x, data.node_to_subgraph)
        #3. 空

        node_embedding = x
        #x = global_add_pool(x, data.subgraph_to_graph)
        x = global_mean_pool(x, data.subgraph_to_graph)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        graph_embedding = x

        if self.task == "type":
            return F.log_softmax(self.lin_graph_clasify(graph_embedding), dim=-1)
        elif self.task == "FA":
            return self.lin_graph_regression(graph_embedding)
        elif self.task in ["0", "1", "2", "3", "4", "5", "6"]:
            return F.log_softmax(self.lin_node_type(node_embedding), dim=-1)

    def __repr__(self):
        return self.__class__.__name__

# class NestedGIN(torch.nn.Module):
#     def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False):
#         super(NestedGIN, self).__init__()
#         self.use_rd = use_rd
#         self.use_z = use_z
#         if self.use_rd:
#             self.rd_projection = torch.nn.Linear(1, 8)
#         if self.use_z:
#             self.z_embedding = torch.nn.Embedding(1000, 8)
#         input_dim = dataset.num_features
#         if self.use_z or self.use_rd:
#             input_dim += 8

#         self.conv1 = GINConv(
#             Sequential(
#                 Linear(input_dim, hidden),
#                 BN(hidden),
#                 ReLU(),
#                 Linear(hidden, hidden),
#                 BN(hidden),
#                 ReLU(),
#             ),
#             train_eps=True)
#         self.convs = torch.nn.ModuleList()
#         for i in range(num_layers - 1):
#             self.convs.append(
#                 GINConv(
#                     Sequential(
#                         Linear(hidden, hidden),
#                         BN(hidden), 
#                         ReLU(),
#                         Linear(hidden, hidden),
#                         BN(hidden), 
#                         ReLU(),
#                     ),
#                     train_eps=True))
#         self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
#         self.lin2 = Linear(hidden, dataset.num_classes)

#         self.task = args.task
#         self.lin_graph_clasify = Linear(hidden, dataset.type.max().item()+1)
#         self.lin_graph_regression = Linear(hidden, 1)
#         self.lin_node_type = Linear(128, 2)  # ??


#     def reset_parameters(self):
#         if self.use_rd:
#             self.rd_projection.reset_parameters()
#         if self.use_z:
#             self.z_embedding.reset_parameters()
#         self.conv1.reset_parameters()
#         for conv in self.convs:
#             conv.reset_parameters()
#         self.lin1.reset_parameters()
#         self.lin2.reset_parameters()

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         # 添加SPD等结点级标签表示
#         # node label embedding
#         z_emb = 0
#         if self.use_z and 'z' in data:
#             ### computing input node embedding
#             z_emb = self.z_embedding(data.z)
#             if z_emb.ndim == 3:
#                 z_emb = z_emb.sum(dim=1)
        
#         if self.use_rd and 'rd' in data:
#             rd_proj = self.rd_projection(data.rd)
#             z_emb += rd_proj

#         if self.use_rd or self.use_z:
#             x = torch.cat([z_emb, x], -1)

#         # 过GNN部分设计
#         x = self.conv1(x, edge_index)
#         xs = [x]
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             xs += [x]

#         x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)

#         node_embedding = x

#         # 获取图界别表示
#         #x = global_add_pool(x, data.subgraph_to_graph)
#         x = global_mean_pool(x, data.subgraph_to_graph)
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=0.5, training=self.training)

#         graph_embedding = x

#         # x = self.lin2(x)

#         # 任务转化器
#         if self.task == "type":
#             return F.log_softmax(self.lin_graph_clasify(graph_embedding), dim=-1)
#         elif self.task == "FA":
#             return self.lin_graph_regression(graph_embedding)
#         elif self.task in ["0", "1", "2", "3", "4", "5", "6"]:
#             return F.log_softmax(self.lin_node_type(node_embedding), dim=-1)

#     def __repr__(self):
#         return self.__class__.__name__