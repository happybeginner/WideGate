import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
import pdb
from torch_geometric.nn.conv.dir_gnn_conv import DirGNNConv

class NestedTwo_DIR_GCN(torch.nn.Module):
    def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False, without_subgraph=False):
        super(NestedTwo_DIR_GCN, self).__init__()
        without_subgraph = True
        self.without_subgraph = without_subgraph
        self.use_rd = use_rd
        self.use_z = use_z
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset.num_features

        if not self.without_subgraph and (self.use_z or self.use_rd):
            input_dim += 8

        # self.conv1 = DirGNNConv(GCNConv(input_dim, hidden))
        self.conv1 = GCNConv(input_dim, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            # self.convs.append(DirGNNConv(GCNConv(hidden, hidden)))
            self.convs.append(GCNConv(hidden, hidden))

        self.conv2 = GCNConv(input_dim, hidden)
        self.convs2 = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs2.append(GCNConv(hidden, hidden))

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
        if self.without_subgraph:
            x, edge_index, batch = data.x_ori, data.original_edge_index, data.batch
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

        x_copy = x

        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]

        x2 = F.relu(self.conv2(x_copy, edge_index.flip([0])))
        xs2 = [x2]
        for conv in self.convs2:
            x2 = F.relu(conv(x2, edge_index.flip([0])))
            xs2 += [x2]


        if not self.without_subgraph:
            x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
        else:
            x = torch.cat(xs, dim=1)
            x2 = torch.cat(xs2, dim=1)

            x = 0.5 * x + 0.5 * x2

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

class NestedDIR_GCN(torch.nn.Module):
    def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False, without_subgraph=False):
        super(NestedDIR_GCN, self).__init__()
        without_subgraph = True
        self.without_subgraph = without_subgraph
        self.use_rd = use_rd
        self.use_z = use_z
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset.num_features

        if not self.without_subgraph and (self.use_z or self.use_rd):
            input_dim += 8

        self.conv1 = DirGNNConv(GCNConv(input_dim, hidden))
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(DirGNNConv(GCNConv(hidden, hidden)))
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
        if self.without_subgraph:
            x, edge_index, batch = data.x_ori, data.original_edge_index, data.batch
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

class NestedGCN(torch.nn.Module):
    def __init__(self, args, dataset, num_layers, hidden, use_z=False, use_rd=False, without_subgraph=True):
        super(NestedGCN, self).__init__()
        self.without_subgraph = without_subgraph
        self.use_rd = use_rd
        self.use_z = use_z
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset.num_features
        if not without_subgraph and (self.use_z or self.use_rd):
            input_dim += 8

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
        if self.without_subgraph:
            x, edge_index, batch = data.x_ori, data.original_edge_index, data.batch
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

        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]

        if self.without_subgraph:
            x = torch.cat(xs, dim=1)
        else:
            x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)

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


class GCN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, *args, **kwargs):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]
        x = global_mean_pool(torch.cat(xs, dim=1), batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
