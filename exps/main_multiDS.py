import os, sys
import argparse
import random
import copy

import torch
from tensorboardX import SummaryWriter
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import setupGC
from training import *


def process_selftrain(args, clients, server, local_epoch):
    print("Self-training ...")
    df = pd.DataFrame()
    allAccs = run_selftrain_GC(args, clients, server, local_epoch)
    for k, v in allAccs.items():
        df.loc[k, [f'train_acc', f'val_acc', f'test_acc']] = v
    print(df)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_'+args.alg+'_GC.csv')
    else:
        outfile = os.path.join(outpath, f'{args.repeat}_accuracy_'+args.alg+'_GC.csv')
    df.to_csv(outfile)
    print(f"Wrote to file: {outfile}")

def process_fedstar(args, clients, server, summary_writer):
    print("\nDone setting up FedStar devices.")

    print("Running FedStar ...")
    frame = run_fedstar(args, clients, server, args.num_rounds, args.local_epoch, samp=None, summary_writer=summary_writer)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedstar_{args.type_init}_GC.csv')
    else:
        outfile = os.path.join(outpath, f'{args.repeat}_accuracy_fedstar_{args.type_init}_GC.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='CPU / GPU device.')
    parser.add_argument('--alg', type=str, default='fedstar',
                        help='Name of algorithms.')
    parser.add_argument('--num_rounds', type=int, default=200,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for inner solver;')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of GINconv layers')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for node classification.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=1)

    parser.add_argument('--datapath', type=str, default='./Data',
                        help='The input path of data.')
    parser.add_argument('--outbase', type=str, default='./outputs',
                        help='The base path for outputting.')
    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default='chem', choices=['chem', 'biochem', 'biochemsn', 'biosncv'])
    parser.add_argument('--seq_length', help='the length of the gradient norm sequence',
                        type=int, default=5)
    parser.add_argument('--n_rw', type=int, default=16,
                        help='Size of position encoding (random walk).')
    parser.add_argument('--n_dg', type=int, default=16,
                        help='Size of position encoding (max degree).')
    parser.add_argument('--n_ones', type=int, default=16,
                        help='Size of position encoding (ones).')
    parser.add_argument('--type_init', help='the type of positional initialization',
                        type=str, default='rw_dg', choices=['rw', 'dg', 'rw_dg', 'ones'])

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    # set seeds
    seed_dataSplit = 123
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # set device
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # set output path
    outpath = os.path.join(args.outbase, 'raw', args.data_group)
    Path(outpath).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {outpath}")

    # preparing data
    print("Preparing data ...")
    splitedData, df_stats = setupGC.prepareData_multiDS(args, args.datapath, args.data_group, args.batch_size, seed=seed_dataSplit)
    print("Done")

    # save statistics of data on clients
    if args.repeat is None:
        outf = os.path.join(outpath, 'stats_trainData.csv')
    else:
        outf = os.path.join(outpath, f'{args.repeat}_stats_trainData.csv')
    df_stats.to_csv(outf)
    print(f"Wrote to {outf}")

    args.n_se = args.n_rw + args.n_dg

    init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData, args)
    print("\nDone setting up devices.")

    # set summarywriter
    if 'fedstar' in args.alg:
        sw_path = os.path.join(args.outbase, 'raw', 'tensorboard', f'{args.data_group}_{args.alg}_{args.type_init}_{args.repeat}')
    else:
        sw_path = os.path.join(args.outbase, 'raw', 'tensorboard', f'{args.data_group}_{args.alg}_{args.repeat}')
    summary_writer = SummaryWriter(sw_path)

    if args.alg == 'selftrain':
        process_selftrain(args, clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), local_epoch=200)
    elif args.alg == 'fedstar':
        process_fedstar(args, clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), summary_writer=summary_writer)