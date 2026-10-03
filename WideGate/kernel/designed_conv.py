import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import RGCNConv, GATConv, GINConv, GCNConv, SAGEConv, global_mean_pool
import pdb
# from torch_geometric.nn.conv.dir_gnn_conv import DirGNNConv

import copy as cp
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

import typing
from typing import Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import PairTensor  # noqa
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    SparseTensor,
)
from torch_geometric.utils import is_torch_sparse_tensor
from torch_geometric.utils.sparse import set_sparse_value

from tqdm import tqdm

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload

from torch_geometric.nn.inits import zeros
from torch import nn

class Designed_FAConv(MessagePassing):
    r"""The Frequency Adaptive Graph Convolution operator from the
    `"Beyond Low-Frequency Information in Graph Convolutional Networks"
    <https://arxiv.org/abs/2101.00797>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i= \epsilon \cdot \mathbf{x}^{(0)}_i +
        \sum_{j \in \mathcal{N}(i)} \frac{\alpha_{i,j}}{\sqrt{d_i d_j}}
        \mathbf{x}_{j}

    where :math:`\mathbf{x}^{(0)}_i` and :math:`d_i` denote the initial feature
    representation and node degree of node :math:`i`, respectively.
    The attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \mathbf{\alpha}_{i,j} = \textrm{tanh}(\mathbf{a}^{\top}[\mathbf{x}_i,
        \mathbf{x}_j])

    based on the trainable parameter vector :math:`\mathbf{a}`.

    Args:
        channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        eps (float, optional): :math:`\epsilon`-value. (default: :obj:`0.1`)
        dropout (float, optional): Dropout probability of the normalized
            coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`).
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\sqrt{d_i d_j}` on first execution, and
            will use the cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops (if
            :obj:`add_self_loops` is :obj:`True`) and compute
            symmetric normalization coefficients on the fly.
            If set to :obj:`False`, :obj:`edge_weight` needs to be provided in
            the layer's :meth:`forward` method. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          initial node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)` or
          :math:`((|\mathcal{V}|, F), ((2, |\mathcal{E}|),
          (|\mathcal{E}|)))` if :obj:`return_attention_weights=True`
    """
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]
    _alpha: OptTensor

    def __init__(self, channels: int, eps: float = 0.1, dropout: float = 0.0,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, bias: bool = False, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.channels = channels
        self.eps = eps
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None
        self._alpha = None

        self.lin = Linear(channels, channels, bias=False,
                          weight_initializer='glorot')

        self.att_l = Linear(channels, 1, bias=False)
        self.att_r = Linear(channels, 1, bias=False)

        # bias 设置
        if bias:
            self.bias = Parameter(torch.empty(channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.att_l.reset_parameters()
        self.att_r.reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    @overload
    def forward(
        self,
        x: Tensor,
        x_0: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Tensor,
        x_0: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Tensor,
        x_0: Tensor,
        edge_index: SparseTensor,
        edge_weight: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass

    def forward(  # noqa: F811
        self,
        x: Tensor,
        x_0: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): The node features.
            x_0 (torch.Tensor): The initial input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        if self.normalize:
            if isinstance(edge_index, Tensor):
                assert edge_weight is None
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, None, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                assert not edge_index.has_value()
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, None, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        else:
            if isinstance(edge_index,
                          Tensor) and not is_torch_sparse_tensor(edge_index):
                assert edge_weight is not None
            elif isinstance(edge_index, SparseTensor):
                assert edge_index.has_value()

        alpha_l = self.att_l(x)
        alpha_r = self.att_r(x)

        # propagate_type: (x: Tensor, alpha: PairTensor,
        #                  edge_weight: OptTensor)
        x = self.lin(x)

        out = self.propagate(edge_index, x=x, alpha=(alpha_l, alpha_r),
                             edge_weight=edge_weight)

        alpha = self._alpha
        self._alpha = None

        if self.eps != 0.0:
            out = out + self.eps * x_0

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            # 自己加的
            if self.bias is not None:
                out = out + self.bias

            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: Tensor,
                edge_weight: OptTensor) -> Tensor:
        assert edge_weight is not None
        alpha = (alpha_j + alpha_i).tanh().squeeze(-1)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * (alpha * edge_weight).view(-1, 1)
        # return x_j * (edge_weight).view(-1, 1)
    
    # def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
    #     return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, eps={self.eps})'


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

class DirGNNConv_RGCN(torch.nn.Module):
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

        self.conv_in = cp.deepcopy(conv)
        self.conv_out = cp.deepcopy(conv)

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

class DirGNNConv_FA(torch.nn.Module):
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

        self.conv_in = cp.deepcopy(conv)
        self.conv_out = cp.deepcopy(conv)

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

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Tensor) -> Tensor:
        """"""  # noqa: D419
        x_in = self.conv_in(x, x_0, edge_index)
        x_out = self.conv_out(x, x_0, edge_index.flip([0]))

        out = self.alpha * x_out + (1 - self.alpha) * x_in

        if self.root_weight:
            out = out + self.lin(x)

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.conv_in}, alpha={self.alpha})'

class NestedDIR_Designed(torch.nn.Module):
    def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False, without_subgraph=True, eps = 0.1):
        super(NestedDIR_Designed, self).__init__()
        self.without_subgraph = without_subgraph
        self.use_rd = use_rd
        self.use_z = use_z
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset[0].num_features

        if not self.without_subgraph and (self.use_z or self.use_rd):
            input_dim += 8

        self.lin_pre = Linear(input_dim, hidden)
        self.conv1 = DirGNNConv_FA(Designed_FAConv(hidden, eps = eps))
        # self.conv1 = Designed_FAConv(hidden, eps = eps)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(DirGNNConv_FA(Designed_FAConv(hidden, eps=eps)))
            # self.convs.append(Designed_FAConv(hidden, eps=eps))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)

        self.task = args.task
        # self.lin_graph_clasify = Linear(hidden, dataset.type.max().item()+1)
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
        if self.without_subgraph:
            x, edge_index, batch = data.x_ori, data.original_edge_index, data.batch
            # x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        
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

        x = self.lin_pre(x)
        x_0 = x
        x = F.relu(self.conv1(x, x_0, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, x_0, edge_index))
            xs += [x]

        if not self.without_subgraph:
            x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
        else:
            x = torch.cat(xs, dim=1)

        node_embedding = x
        #x = global_add_pool(x, data.subgraph_to_graph)
        # x = global_mean_pool(x, data.subgraph_to_graph)
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

class NestedDIR_SAGE_Designed(torch.nn.Module):
    def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False, without_subgraph=True, eps = 0.1):
        super(NestedDIR_SAGE_Designed, self).__init__()
        self.without_subgraph = without_subgraph
        self.use_rd = use_rd
        self.use_z = use_z
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset[0].num_features

        if not self.without_subgraph and (self.use_z or self.use_rd):
            input_dim += 8

        self.lin_pre = Linear(input_dim, hidden)
        # self.conv1 = DirGNNConv(SAGEConv(input_dim, hidden))
        # self.conv1 = DirGNNConv(SAGEConv(input_dim, hidden))
        self.conv1 = SAGEConv(input_dim, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            # self.convs.append(DirGNNConv(SAGEConv(hidden, hidden)))
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)

        self.task = args.task
        # self.lin_graph_clasify = Linear(hidden, dataset.type.max().item()+1)
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
        if self.without_subgraph:
            x, edge_index, batch = data.x_ori, data.original_edge_index, data.batch
            # x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        
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

        # x = self.lin_pre(x)
        # x_0 = x
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]

        if not self.without_subgraph:
            x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
        else:
            x = torch.cat(xs, dim=1)

        node_embedding = x
        #x = global_add_pool(x, data.subgraph_to_graph)
        # x = global_mean_pool(x, data.subgraph_to_graph)
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

class NestedDIR_RGCN_Designed(torch.nn.Module):
    def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False, without_subgraph=True, num_relations=2):
        super(NestedDIR_RGCN_Designed, self).__init__()
        self.without_subgraph = without_subgraph
        self.use_rd = use_rd
        self.use_z = use_z
        self.args = args
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset[0].num_features

        if not self.without_subgraph and (self.use_z or self.use_rd):
            input_dim += 8

        # self.lin_pre = Linear(input_dim, hidden)
        self.embedding = nn.Linear(input_dim, out_features= hidden, bias=False)
        # self.conv1 = DirGNNConv(Designed_FAConv(hidden, eps = eps))
        self.conv1 = DirGNNConv_RGCN(RGCNConv(input_dim, hidden, num_relations))
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(DirGNNConv_RGCN(RGCNConv(hidden, hidden, num_relations)))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)

        self.task = args.task
        # self.lin_graph_clasify = Linear(hidden, dataset.type.max().item()+1)
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
        if self.without_subgraph:
            x, edge_index, batch = data.x_ori, data.original_edge_index, data.batch
            edge_types = data.edge_type
            # x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        
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

        # x = self.lin_pre(x)
        if self.args.pe != "None":
            x = self.embedding(x)
        # x_0 = x
        x = F.relu(self.conv1(x, edge_index, edge_types))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_types))
            xs += [x]

        if not self.without_subgraph:
            x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
        else:
            x = torch.cat(xs, dim=1)

        node_embedding = x
        #x = global_add_pool(x, data.subgraph_to_graph)
        # x = global_mean_pool(x, data.subgraph_to_graph)
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

class NestedDIR_SAGE_Designed(torch.nn.Module):
    def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False, without_subgraph=True, eps = 0.1):
        super(NestedDIR_SAGE_Designed, self).__init__()
        self.without_subgraph = without_subgraph
        self.use_rd = use_rd
        self.use_z = use_z
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset[0].num_features

        if not self.without_subgraph and (self.use_z or self.use_rd):
            input_dim += 8

        self.lin_pre = Linear(input_dim, hidden)
        # self.conv1 = DirGNNConv(SAGEConv(input_dim, hidden))
        # self.conv1 = DirGNNConv(SAGEConv(input_dim, hidden))
        self.conv1 = SAGEConv(input_dim, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            # self.convs.append(DirGNNConv(SAGEConv(hidden, hidden)))
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)

        self.task = args.task
        # self.lin_graph_clasify = Linear(hidden, dataset.type.max().item()+1)
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
        if self.without_subgraph:
            x, edge_index, batch = data.x_ori, data.original_edge_index, data.batch
            # x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        
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

        # x = self.lin_pre(x)
        # x_0 = x
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]

        if not self.without_subgraph:
            x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
        else:
            x = torch.cat(xs, dim=1)
            

        node_embedding = x
        #x = global_add_pool(x, data.subgraph_to_graph)
        # x = global_mean_pool(x, data.subgraph_to_graph)
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

class NestedDIR_GAT_Designed(torch.nn.Module):
    def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False, without_subgraph=True, eps = 0.1):
        super(NestedDIR_GAT_Designed, self).__init__()
        self.without_subgraph = without_subgraph
        self.use_rd = use_rd
        self.use_z = use_z
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset[0].num_features

        if not self.without_subgraph and (self.use_z or self.use_rd):
            input_dim += 8

        self.lin_pre = Linear(input_dim, hidden)
        # self.conv1 = DirGNNConv(SAGEConv(input_dim, hidden))
        # self.conv1 = DirGNNConv(SAGEConv(input_dim, hidden))
        self.conv1 = GATConv(input_dim, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            # self.convs.append(DirGNNConv(SAGEConv(hidden, hidden)))
            self.convs.append(GATConv(hidden, hidden))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)

        self.task = args.task
        # self.lin_graph_clasify = Linear(hidden, dataset.type.max().item()+1)
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
        if self.without_subgraph:
            x, edge_index, batch = data.x_ori, data.original_edge_index, data.batch
            # x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        
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

        # x = self.lin_pre(x)
        # x_0 = x
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]

        if not self.without_subgraph:
            x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
        else:
            x = torch.cat(xs, dim=1)
            

        node_embedding = x
        #x = global_add_pool(x, data.subgraph_to_graph)
        # x = global_mean_pool(x, data.subgraph_to_graph)
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

class NestedDIR_GIN_Designed(torch.nn.Module):
    def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False, without_subgraph=True, eps = 0.1):
        super(NestedDIR_GIN_Designed, self).__init__()
        self.without_subgraph = without_subgraph
        self.use_rd = use_rd
        self.use_z = use_z
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset[0].num_features

        if not self.without_subgraph and (self.use_z or self.use_rd):
            input_dim += 8

        self.lin_pre = Linear(input_dim, hidden)
        # self.conv1 = DirGNNConv(SAGEConv(input_dim, hidden))
        # self.conv1 = DirGNNConv(SAGEConv(input_dim, hidden))
        self.conv1 = GINConv(input_dim, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            # self.convs.append(DirGNNConv(SAGEConv(hidden, hidden)))
            self.convs.append(GINConv(hidden, hidden))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)

        self.task = args.task
        # self.lin_graph_clasify = Linear(hidden, dataset.type.max().item()+1)
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
        if self.without_subgraph:
            x, edge_index, batch = data.x_ori, data.original_edge_index, data.batch
            # x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        
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

        # x = self.lin_pre(x)
        # x_0 = x
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]

        if not self.without_subgraph:
            x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
        else:
            x = torch.cat(xs, dim=1)
            

        node_embedding = x
        #x = global_add_pool(x, data.subgraph_to_graph)
        # x = global_mean_pool(x, data.subgraph_to_graph)
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

class SAGE_MULT(torch.nn.Module):
    # def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
    #              dropout):
    def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False, without_subgraph=True, eps = 0.1):
        super(SAGE_MULT, self).__init__()
        self.num_layers = num_layers


        in_channels = dataset[0].num_features
        self.hidden_channels = hidden
        hidden_channels = hidden
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, 'mean'))
        # self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, 'mean'))
            # self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, 'mean'))
        # self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # two linear layer for predictions
        self.linear = torch.nn.ModuleList()
        self.linear.append(Linear(hidden_channels, hidden_channels, bias=False))
        self.linear.append(Linear(hidden_channels, 2, bias=False))
        # self.linear.append(Linear(hidden_channels, out_channels, bias=False))
        # self.linear.append(Linear(hidden_channels, out_channels, bias=False))
        
        self.bn0 = BatchNorm1d(hidden_channels)

        # self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.linear:
            lin.reset_parameters()
    
    def forward(self, data):
        # tensor placement
        # h = x
        x, edge_index, batch = data.x_ori, data.original_edge_index, data.batch
        graph = edge_index
        h = x    
        
        for conv in self.convs:
            # h = conv(graph, h)
            h = conv(h, graph)
            h = F.relu(h)
            h = F.dropout(h, p=0.5, training=self.training)

        # print(x[0])
        h = self.linear[0](h)
        h = self.bn0(F.relu(h))
        # x = F.dropout(x, p=0.5, training=self.training)
        x1 = self.linear[1](h) # for xor

        # print(self.linear[0].weight)
        # print(x1[0])
        return F.log_softmax(x1, dim=-1)

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

def homo_2_hyter_TWO(dataset):
    new_dataset = []
    for data in tqdm(dataset):
        # hetero_data = HeteroData()
        node_types = torch.argmax(data.x[:, :3], dim=1) 
        edge_index = data.edge_index
        
        # for node_type in torch.unique(node_types):
        #     hetero_data[f'node_type_{node_type.item()}'].x = data.x[node_types == node_type]

        # edge_class = [
        #     (0, 0),
        #     (0, 1),  # node_type_0 -> node_type_1
        #     (0, 2),  # node_type_0 -> node_type_2
        #     (1, 1),
        #     (1, 0),  # node_type_1 -> node_type_0
        #     (1, 2),  # node_type_1 -> node_type_2
        #     (2, 2),
        #     (2, 0),  # node_type_2 -> node_type_0
        #     (2, 1),  # node_type_2 -> node_type_1
        # ]

        edge_types = torch.empty(edge_index.size(1), dtype=torch.long)
        # for i, (src, dst) in enumerate(edge_class):
        #     src_type = src
        #     dst_type = dst

        for j, (edge_src, edge_dst) in enumerate(edge_index.T):
            if node_types[edge_src.item()] == 2 or node_types[edge_dst.item()] == 2:
                edge_types[j] = 0     # 过not门的边
            else:
                edge_types[j] = 1     # 不过not门的边
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