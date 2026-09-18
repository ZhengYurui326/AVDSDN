import pickle

import numpy as np
from vmdpy import VMD


def process_with_vmd(input_path, output_path):
    with open(input_path, 'rb') as f:
        Xd = pickle.load(f, encoding='iso-8859-1')

    Xd_vmd = {}

    ALPHA = 10
    TAU = 0.0
    K = 5
    DC = 0
    INIT = 2
    TOL = 1e-7

    j = 0

    for key in Xd.keys():
        raw_data = Xd[key]
        num_samples = raw_data.shape[0]

        processed_data = np.zeros((num_samples, K + 1, 2, 128), dtype=np.float32)

        for i in range(num_samples):
            I_signal = raw_data[i, 0, :].astype(np.float32)
            Q_signal = raw_data[i, 1, :].astype(np.float32)

            u_I, _, _ = VMD(I_signal, ALPHA, TAU, K, DC, INIT, TOL)
            u_Q, _, _ = VMD(Q_signal, ALPHA, TAU, K, DC, INIT, TOL)

            u_I = u_I.astype(np.float32)
            u_Q = u_Q.astype(np.float32)

            for k in range(K):
                processed_data[i, k, 0, :] = u_I[k]
                processed_data[i, k, 1, :] = u_Q[k]

            processed_data[i, K, 0, :] = I_signal.astype(np.float32)
            processed_data[i, K, 1, :] = Q_signal.astype(np.float32)

        Xd_vmd[key] = processed_data

        j = j + 1
        print(key, j)

    with open(output_path, 'wb') as f:
        pickle.dump(Xd_vmd, f)


def load_data_vmd(seed=3407, filename='dataset/RML2016.10a_vmd_float32.pkl'):
    Xd = pickle.load(open(filename, 'rb'), encoding='iso-8859-1')
    mods, snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0, 1]]

    X = []
    lbl = []
    train_idx = []
    val_idx = []
    np.random.seed(seed)
    a = 0

    for mod in mods:
        for snr in snrs:
            data = Xd[(mod, snr)][:, :5, :, :]
            X.append(data)

            lbl.extend([(mod, snr)] * data.shape[0])

            train_idx += list(
                np.random.choice(
                    range(a * 1000, (a + 1) * 1000),
                    size=600,
                    replace=False
                )
            )

            val_idx += list(
                np.random.choice(
                    list(
                        set(range(a * 1000, (a + 1) * 1000))
                        - set(train_idx[-600:])
                    ),
                    size=200,
                    replace=False
                )
            )
            a += 1

    X = np.vstack(X)

    n_examples = X.shape[0]
    test_idx = list(
        set(range(0, n_examples)) - set(train_idx) - set(val_idx)
    )
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    def to_onehot(yy):
        yy1 = np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_val = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    print("Y_train", Y_train.shape)
    print("Y_val", Y_val.shape)
    print("Y_test", Y_test.shape)

    return (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx)
