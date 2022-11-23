import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_selftrain_GC(args, clients, server, local_epoch):
    # all clients are initialized with the same weights
    for client in clients:
        client.download_from_server(args, server)

    allAccs = {}
    for client in clients:
        client.local_train(local_epoch)

        loss, acc = client.evaluate()
        allAccs[client.name] = [client.train_stats['trainingAccs'][-1], client.train_stats['valAccs'][-1], acc]
        print("  > {} done.".format(client.name))

    return allAccs


def run_fedavg(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0, summary_writer=None):
    for client in clients:
        client.download_from_server(args, server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            # only get weights of graphconv layers
            client.local_train(local_epoch)

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(args, server)

        # write to log files
        if c_round % 5 == 0:
            for idx in range(len(clients)):
                loss, acc = clients[idx].evaluate()
                summary_writer.add_scalar('Test/Acc/user' + str(idx + 1), acc, c_round)
                summary_writer.add_scalar('Test/Loss/user' + str(idx + 1), loss, c_round)

    frame = pd.DataFrame()
    for client in clients:
        loss, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame


def run_fedstar(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0, summary_writer=None):
    for client in clients:
        client.download_from_server(args, server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            # only get weights of graphconv layers
            client.local_train(local_epoch)

        server.aggregate_weights_se(selected_clients)
        for client in selected_clients:
            client.download_from_server(args, server)

        # write to log files
        if c_round % 5 == 0:
            for idx in range(len(clients)):
                loss, acc = clients[idx].evaluate()
                summary_writer.add_scalar('Test/Acc/user' + str(idx + 1), acc, c_round)
                summary_writer.add_scalar('Test/Loss/user' + str(idx + 1), loss, c_round)

    frame = pd.DataFrame()
    for client in clients:
        loss, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame


