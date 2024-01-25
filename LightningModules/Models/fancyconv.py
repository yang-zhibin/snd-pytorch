from multiprocessing.sharedctypes import Value
import torch.nn as nn
import torch
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_std, scatter_mean
from torch_geometric.nn import knn_graph, radius_graph
from torch.nn import functional as F

from ..gnn_base import GNNBase
from ..utils import make_mlp


class FancyConv(nn.Module):
    def __init__(self, hparams, input_size=None, output_size=None):
        super().__init__()
        self.hparams = hparams
        self.feature_dropout = hparams.get("feature_dropout", 0.0)
        self.spatial_dropout = hparams.get("spatial_dropout", 0.0)
        self.conv_dropout = hparams.get("conv_dropout", 0.0)
        self.input_size = hparams["hidden"] if input_size is None else input_size
        self.output_size = hparams["hidden"] if output_size is None else output_size
        
        # number of aggregators
        self.aggs = hparams.get("aggs", ['add']) # we don't edit the list
        self.n_agg = len(self.aggs)

        self.agg_func = {'add': scatter_add, 'max': scatter_max, 'min':scatter_min, 'std':scatter_std, 'mean':scatter_mean}

        self.feature_network = make_mlp(
                (1+self.n_agg)*(self.input_size) + 1,
                [self.output_size] * hparams["nb_feature_layer"],
                output_activation=hparams["hidden_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                dropout=self.feature_dropout
        )

        self.spatial_network = make_mlp(
                self.input_size + 1,
                [self.input_size] * hparams["nb_spatial_layer"] + [hparams["emb_dims"]],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                dropout=self.spatial_dropout
        )

        self.conv_network = make_mlp(
            # (self.output_size + 1)*2, # Gets the output of the feature network for the node, and the difference vector for the other node
            # [self.output_size]*hparams["n_conv_layer"],
            (self.input_size + 1)*2, # Gets the output of the feature network for the node, and the difference vector for the other node
            [self.input_size]*hparams["n_conv_layer"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=True,
            batch_norm=False,
            dropout=self.conv_dropout
        )

        # This handles the various r, k, and random edge options
        self.setup_neighborhood_configuration()

    def get_neighbors(self, spatial_features):
        
        edge_index = torch.empty([2, 0], dtype=torch.int64, device=spatial_features.device)
 
        if self.use_radius:
            radius_edges = radius_graph(spatial_features, r=self.r, max_num_neighbors=self.hparams["max_knn"], batch=self.batch, loop=self.hparams["self_loop"])
            edge_index = torch.cat([edge_index, radius_edges], dim=1)
        
        # Remove duplicate edges
        edge_index = torch.unique(edge_index, dim=1)

        return edge_index

    def get_grav_function(self, d):
        grav_weight = self.grav_weight
        grav_function = - grav_weight * d / self.r**2
        
        return grav_function

    def get_attention_weight(self, spatial_features, edge_index):
        start, end = edge_index
        d = torch.sum((spatial_features[start] - spatial_features[end])**2, dim=-1) 
        grav_function = self.get_grav_function(d)

        return torch.exp(grav_function)

    def grav_pooling(self, spatial_features, fts):
        edge_index = self.get_neighbors(spatial_features)
        start, end = edge_index

        if "norm_hidden" in self.hparams and self.hparams["norm_hidden"]:
            fts = F.normalize(fts, p=1, dim=-1)

        # Subtract the values at the end indices
        x = fts[start]
        x1 = x - fts[end]

        x = torch.stack([x, x1], dim=2).view(-1, 2*x.shape[1]) # interleave the features
        x = self.conv_network(x)

        if self.hparams['use_attention_weight']:
            d_weight = self.get_attention_weight(spatial_features, edge_index)
            x = x*d_weight.unsqueeze(1)

        agg_hidden = None
        for agg in self.aggs:
            the_agg = self.agg_func[agg](x, end, dim=0, dim_size=fts.shape[0])
            if isinstance(the_agg, tuple): # scatter_max/min return a tuple but we're not interested in the max/min indices
                the_agg = the_agg[0]
            agg_hidden = the_agg if agg_hidden is None else torch.cat([agg_hidden, the_agg], dim=-1)

        # if torch.isnan(agg_hidden).any():
        #     print('WARNING, Nan value found after scatters')
    
        return agg_hidden, edge_index

    def forward(self, hidden_features, batch, current_epoch):
        self.current_epoch = current_epoch
        self.batch = batch

        hidden_features = torch.cat([hidden_features, hidden_features.mean(dim=1).unsqueeze(-1)], dim=-1)
        spatial_features = self.spatial_network(hidden_features)

        if "norm_embedding" in self.hparams and self.hparams["norm_embedding"]:
            spatial_features = F.normalize(spatial_features, p=2, dim=-1)

        aggregated_hidden, edge_index = self.grav_pooling(spatial_features, hidden_features)
        concatenated_hidden = torch.cat([aggregated_hidden, hidden_features], dim=-1)

        return self.feature_network(concatenated_hidden), edge_index

    def setup_neighborhood_configuration(self):
        self.current_epoch = 0
        self.use_radius = bool("r" in self.hparams and self.hparams["r"])
        # A fix here for the case where there is dropout and a large embedded space, model initially can't find neighbors: Enforce self-loop
        if not self.hparams["knn"] and self.hparams["emb_dims"] > 4 and (self.hparams["feature_dropout"] or self.hparams["spatial_dropout"]):
            self.hparams["self_loop"] = True
        self.use_knn = bool("knn" in self.hparams and self.hparams["knn"])
        self.use_rand_k = bool("rand_k" in self.hparams and self.hparams["rand_k"])

    @property
    def r(self):
        if isinstance(self.hparams["r"], list):
            if len(self.hparams["r"]) == 2:
                return self.hparams["r"][0] + ( (self.hparams["r"][1] - self.hparams["r"][0]) * self.current_epoch / self.hparams["max_epochs"] )
            elif len(self.hparams["r"]) == 3:
                if self.current_epoch < self.hparams["max_epochs"]/2:
                    return self.hparams["r"][0] + ( (self.hparams["r"][1] - self.hparams["r"][0]) * self.current_epoch / (self.hparams["max_epochs"]/2) )
                else:
                    return self.hparams["r"][1] + ( (self.hparams["r"][2] - self.hparams["r"][1]) * (self.current_epoch - self.hparams["max_epochs"]/2) / (self.hparams["max_epochs"]/2) )
        elif isinstance(self.hparams["r"], float):
            return self.hparams["r"]
        else:
            return 0.3

    @property
    def grav_weight(self):        
        if isinstance(self.hparams["grav_weight"], list) and len(self.hparams["grav_weight"]) == 2:
            return (self.hparams["grav_weight"][0] + (self.hparams["grav_weight"][1] - self.hparams["grav_weight"][0]) * self.current_epoch / self.hparams["max_epochs"])
        elif isinstance(self.hparams["grav_weight"], float):
            return self.hparams["grav_weight"]
        else:
            raise ValueError("grav_weight must be a list of length 2 or a float")
        