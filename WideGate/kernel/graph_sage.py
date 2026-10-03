import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, global_mean_pool

from torch_geometric.nn.conv.dir_gnn_conv import DirGNNConv

class NestedDIR_GraphSAGE(torch.nn.Module):
    def __init__(self,args,  dataset, num_layers, hidden, use_z=False, use_rd=False, without_subgraph = True):
        super(NestedDIR_GraphSAGE, self).__init__()
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

        self.conv1 = DirGNNConv(SAGEConv(input_dim, hidden))
        # self.conv1 = SAGEConv(input_dim, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(DirGNNConv(SAGEConv(hidden, hidden)))
            # self.convs.append(SAGEConv(hidden, hidden))
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
        if not self.without_subgraph:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
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

        x = global_mean_pool(x, data.subgraph_to_graph)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin2(x)

        graph_embedding = x

        # x = self.lin2(x)

        if self.task == "type":
            return F.log_softmax(self.lin_graph_clasify(graph_embedding), dim=-1)
        elif self.task == "FA":
            return self.lin_graph_regression(graph_embedding)
        elif self.task in ["0", "1", "2", "3", "4", "5", "6"]:
            return F.log_softmax(self.lin_node_type(node_embedding), dim=-1)

        # return F.log_softmax(x, dim=-1)

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

class NestedGraphSAGE(torch.nn.Module):
    def __init__(self,args,  dataset, num_layers, hidden, use_z=False, use_rd=False):
        super(NestedGraphSAGE, self).__init__()
        self.use_rd = use_rd
        self.use_z = use_z
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset.num_features
        if self.use_z or self.use_rd:
            input_dim += 8

        self.conv1 = SAGEConv(input_dim, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
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

        if self.use_rd or self.use_z:
            x = torch.cat([z_emb, x], -1)

        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]
        
        x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)

        node_embedding = x

        x = global_mean_pool(x, data.subgraph_to_graph)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin2(x)

        graph_embedding = x

        # x = self.lin2(x)

        if self.task == "type":
            return F.log_softmax(self.lin_graph_clasify(graph_embedding), dim=-1)
        elif self.task == "FA":
            return self.lin_graph_regression(graph_embedding)
        elif self.task in ["0", "1", "2", "3", "4", "5", "6"]:
            return F.log_softmax(self.lin_node_type(node_embedding), dim=-1)

        # return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, *args, **kwargs):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
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


class GraphSAGEWithoutJK(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GraphSAGEWithoutJK, self).__init__()
        self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
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
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
