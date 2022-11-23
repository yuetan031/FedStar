import os
import argparse
import pandas as pd
from pathlib import Path

def _aggregate(inpath, outpath, filename):
    dfs = []
    for file in os.listdir(inpath):
        if file.endswith(filename):
            dfs.append(pd.read_csv(os.path.join(inpath, file), header=0, index_col=0))
    df = pd.concat(dfs)
    group = df.groupby(df.index)
    dfmean = group.mean()
    dfstd = group.std()
    df_out = dfmean.join(dfstd, lsuffix='_mean', rsuffix='_std')
    df_out.to_csv(os.path.join(outpath, filename), header=True, index=True)


def average_aggregate_all(outpath):
    algos = ['selftrain', 'fedstar_rw_dg']
    dfs = pd.DataFrame(index=algos, columns=['avg. of test_accuracy_mean', 'avg. of test_accuracy_std'])
    for algo in algos:
        df = pd.read_csv(os.path.join(outpath, f'accuracy_{algo}_GC.csv'), header=0, index_col=0)
        if algo == 'selftrain':
            df = df[['test_acc_mean', 'test_acc_std']]
        dfs.loc[algo] = list(df.mean())
    outfile = os.path.join(outpath, f'avg_accuracy_allAlgos_GC.csv')
    dfs.to_csv(outfile, header=True, index=True)
    print("Wrote to:", outfile)

def main_aggregate_all_multiDS(inpath, outpath):
    """ multiDS: aggregagte all outputs """
    Path(outpath).mkdir(parents=True, exist_ok=True)
    for filename in ['accuracy_selftrain_GC.csv', 'accuracy_fedstar_rw_dg_GC.csv']:
        _aggregate(inpath, outpath, filename)

    """ get average performance for all algorithms """
    average_aggregate_all(outpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default='chem', choices=['chem', 'biochem', 'biochemsn', 'biosncv'])

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    args.inpath = './outputs/raw/' + args.data_group
    args.outpath = './outputs/processed/' + args.data_group

    #     """ multiDS: aggregagte all outputs """
    main_aggregate_all_multiDS(args.inpath, args.outpath)

