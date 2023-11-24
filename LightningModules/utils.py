import os
import warnings
import time
import itertools
import gzip
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch import nn

warnings.simplefilter(action='ignore', category=FutureWarning)

def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation="ReLU",
    layer_norm=False,
    batch_norm=True,
    dropout=0.0,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers with dropout
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(hidden_activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)



def open_processed_files(input_dir, n_events):
    N_EVENTS_PER_FILE = 72800

    files = os.listdir(input_dir)
    num_files = (n_events // N_EVENTS_PER_FILE) + 1 if n_events > 0 else 0
    paths = [os.path.join(input_dir, file) for file in files][:num_files]

    opened_files = [torch.load(file) for file in tqdm(paths)]
    
    opened_files = list(itertools.chain.from_iterable(opened_files))
    
    return opened_files

def load_processed_datasets(input_dir, data_split, graph_construction):
    
    print("Loading torch files")
    print(time.ctime())
    train_events = open_processed_files(os.path.join(input_dir, "train"), data_split[0])
    val_events = open_processed_files(os.path.join(input_dir, "val"), data_split[1])
    test_events = open_processed_files(os.path.join(input_dir, "test"), data_split[2])

    print("Building events")
    print(time.ctime())
    train_dataset = build_processed_dataset(train_events, graph_construction,  data_split[0])
    val_dataset = build_processed_dataset(val_events, graph_construction, data_split[1])
    test_dataset = build_processed_dataset(test_events, graph_construction, data_split[2])


    return train_dataset, val_dataset, test_dataset

def build_processed_dataset(events, graph_construction, n_events=None):
    if n_events == 0:
        return None
    
    subsample = events[:n_events] if n_events is not None else events

    try:
        _ = subsample[0].strip_x
    except Exception:
        print('WARNING, in the funny exception')
        for i, data in enumerate(subsample):
            subsample[i] = Data.from_dict(data.__dict__)

    if (graph_construction == "fully_connected"):        
        for ev in subsample:
            ev.edge_index = get_fully_connected_edges(ev.strip_x)

    print("Testing sample quality")
    empty_rows = []
    for i, sample in enumerate(tqdm(subsample)):
        sample.x = sample.strip_x

        if len(sample.x) == 0:
            empty_rows.append(i)

        # Check if any nan values in sample
        for key in sample.keys():
            bad = torch.isnan(sample[key]).any()
            if bad:
                print('Nan value found in sample in column', key)
                sample[key][torch.isnan(sample[key])] = 0.
                assert not bad, "Nan value found in sample"
    
    print('WARNING: Found', len(empty_rows), 'empty rows')
    for empty_row in reversed(empty_rows):
        subsample = subsample[:empty_row] +subsample[empty_row+1:]

    if len(subsample) < n_events:
        print('WARNING: Subsample loaded with size', len(subsample), 'n events desired', n_events)
    return subsample

"""
Returns an array of edge links corresponding to a fully-connected graph - NEW VERSION
"""
def get_fully_connected_edges(x):
    
    n_nodes = len(x)
    node_list = torch.arange(n_nodes)
    edges = torch.combinations(node_list, r=2).T
    
    return torch.cat([edges, edges.flip(0)], axis=1)
