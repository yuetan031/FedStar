import random
from random import choices
import numpy as np
import pandas as pd

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree

from models import GIN, serverGIN, GIN_dc, serverGIN_dc
from server import Server
from client import Client_GC
from utils import get_maxDegree, get_stats, split_data, get_numGraphLabels, init_structure_encoding

from scipy.special import rel_entr
import scipy
from torch_geometric.utils import erdos_renyi_graph, degree

def _randChunk(graphs, num_client, overlap, seed=None):
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs)
    minSize = min(50, int(totalNum/num_client))
    graphs_chunks = []
    if not overlap:
        for i in range(num_client):
            graphs_chunks.append(graphs[i*minSize:(i+1)*minSize])
        for g in graphs[num_client*minSize:]:
            idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
            graphs_chunks[idx_chunk].append(g)
    else:
        sizes = np.random.randint(low=50, high=150, size=num_client)
        for s in sizes:
            graphs_chunks.append(choices(graphs, k=s))
    return graphs_chunks


def prepareData_oneDS(datapath, data, num_client, batchSize, convert_x=False, seed=None, overlap=False):
    if data == "COLLAB":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
    elif data == "IMDB-BINARY":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
    elif data == "IMDB-MULTI":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
    else:
        tudataset = TUDataset(f"{datapath}/TUDataset", data)
        if convert_x:
            maxdegree = get_maxDegree(tudataset)
            tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
    graphs = [x for x in tudataset]
    print("  **", data, len(graphs))

    graphs_chunks = _randChunk(graphs, num_client, overlap, seed=seed)
    splitedData = {}
    df = pd.DataFrame()
    num_node_features = graphs[0].num_node_features
    for idx, chunks in enumerate(graphs_chunks):
        ds = f'{idx}-{data}'
        ds_tvt = chunks
        ds_train, ds_vt = split_data(ds_tvt, train=0.8, test=0.2, shuffle=True, seed=seed)
        ds_val, ds_test = split_data(ds_vt, train=0.5, test=0.5, shuffle=True, seed=seed)
        dataloader_train = DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(ds_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(ds_test, batch_size=batchSize, shuffle=True)
        num_graph_labels = get_numGraphLabels(ds_train)
        splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                           num_node_features, num_graph_labels, len(ds_train))
        df = get_stats(df, ds, ds_train, graphs_val=ds_val, graphs_test=ds_test)

    return splitedData, df

def prepareData_multiDS(args, datapath, group='chem', batchSize=32, seed=None):
    assert group in ['chem', "biochem", 'biochemsn', "biosncv"]

    if group == 'chem':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"]
    elif group == 'biochem':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS"]                               # bioinformatics
    elif group == 'biochemsn':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS",                               # bioinformatics
                    "COLLAB", "IMDB-BINARY", "IMDB-MULTI"]                     # social networks
    elif group == 'biosncv':
        datasets = ["ENZYMES", "DD", "PROTEINS",                               # bioinformatics
                    "COLLAB", "IMDB-BINARY", "IMDB-MULTI",                     # social networks
                    "Letter-high", "Letter-low", "Letter-med"]                 # computer vision

    splitedData = {}
    df = pd.DataFrame()
    for data in datasets:
        if data == "COLLAB":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
        elif data == "IMDB-BINARY":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
        elif data == "IMDB-MULTI":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
        elif "Letter" in data:
            tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=True)
        else:
            tudataset = TUDataset(f"{datapath}/TUDataset", data)

        graphs = [x for x in tudataset]
        print("  **", data, len(graphs))

        graphs_train, graphs_valtest = split_data(graphs, test=0.2, shuffle=True, seed=seed)
        graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)

        graphs_train = init_structure_encoding(args, gs=graphs_train, type_init=args.type_init)
        graphs_val = init_structure_encoding(args, gs=graphs_val, type_init=args.type_init)
        graphs_test = init_structure_encoding(args, gs=graphs_test, type_init=args.type_init)

        dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)

        num_node_features = graphs[0].num_node_features
        num_graph_labels = get_numGraphLabels(graphs_train)

        splitedData[data] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                             num_node_features, num_graph_labels, len(graphs_train))

        df = get_stats(df, data, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)

    return splitedData, df

def prepareData_multiDS_multi(args, datapath, group='small', batchSize=32, nc_per_ds=1, seed=None):
    assert group in ['chem', "biochem", 'biochemsn', "biosncv"]

    if group == 'chem':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"]
    elif group == 'biochem':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS"]                               # bioinformatics
    elif group == 'biochemsn':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS",                               # bioinformatics
                    "COLLAB", "IMDB-BINARY", "IMDB-MULTI"]                     # social networks
                    # "Letter-low", "Letter-med"]                                # computer vision
    elif group == 'biosncv':
        datasets = ["ENZYMES", "DD", "PROTEINS",                               # bioinformatics
                    "COLLAB", "IMDB-BINARY", "IMDB-MULTI",                     # social networks
                    "Letter-high", "Letter-low", "Letter-med"]                 # computer vision

    splitedData = {}
    df = pd.DataFrame()
    for data in datasets:
        if data == "COLLAB":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
        elif data == "IMDB-BINARY":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
        elif data == "IMDB-MULTI":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
        elif "Letter" in data:
            tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=True)
        else:
            tudataset = TUDataset(f"{datapath}/TUDataset", data)

        graphs = [x for x in tudataset]
        print("  **", data, len(graphs))
        num_node_features = graphs[0].num_node_features
        graphs_chunks = _randChunk(graphs, nc_per_ds, overlap=False, seed=seed)
        for idx, chunks in enumerate(graphs_chunks):
            ds = f'{idx}-{data}'
            ds_tvt = chunks
            graphs_train, graphs_valtest = split_data(ds_tvt, train=0.8, test=0.2, shuffle=True, seed=seed)
            graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)

            graphs_train = init_structure_encoding(args, gs=graphs_train, type_init=args.type_init)
            graphs_val = init_structure_encoding(args, gs=graphs_val, type_init=args.type_init)
            graphs_test = init_structure_encoding(args, gs=graphs_test, type_init=args.type_init)

            dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
            dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
            dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)
            num_graph_labels = get_numGraphLabels(graphs_train)
            splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                               num_node_features, num_graph_labels, len(graphs_train))
            df = get_stats(df, ds, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)

    return splitedData, df

def js_diver(P,Q):
    M=P+Q
    return 0.5*scipy.stats.entropy(P,M,base=2)+0.5*scipy.stats.entropy(Q,M,base=2)

def setup_devices(splitedData, args):
    idx_clients = {}
    clients = []
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
        if args.alg == 'fedstar':
            cmodel_gc = GIN_dc(num_node_features, args.n_se, args.hidden, num_graph_labels, args.nlayer, args.dropout)
        else:
            cmodel_gc = GIN(num_node_features, args.hidden, num_graph_labels, args.nlayer, args.dropout)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel_gc.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        clients.append(Client_GC(cmodel_gc, idx, ds, train_size, dataloaders, optimizer, args))

    if args.alg == 'fedstar':
        smodel = serverGIN_dc(n_se=args.n_se, nlayer=args.nlayer, nhid=args.hidden)
    else:
        smodel = serverGIN(nlayer=args.nlayer, nhid=args.hidden)
    server = Server(smodel, args.device)
    return clients, server, idx_clients
