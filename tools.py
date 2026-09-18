import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def plot_confusion_matrix_big(cm, title='Confusion matrix', cmap=plt.get_cmap("Blues"), labels=[], save_filename=None):
    num_classes = len(labels)

    plt.figure(figsize=(20, 20), dpi=300)

    im = plt.imshow(cm * 100, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=20, pad=20)

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Acc (%)', rotation=270, labelpad=20, fontsize=16)

    tick_marks = np.arange(num_classes)

    font_size = max(6, min(10, 250 / num_classes))

    plt.xticks(tick_marks, labels, rotation=90, ha='center', fontsize=font_size)
    plt.yticks(tick_marks, labels, fontsize=font_size)

    for i in range(num_classes):
        for j in range(num_classes):
            value = int(np.around(cm[i, j] * 100))

            if i == j:
                color = 'red'
                fontweight = 'bold'
            else:
                color = 'black'
                fontweight = 'normal'

            plt.text(j, i, f'{value}%',
                     ha="center",
                     va="center",
                     color=color,
                     fontsize=font_size - 2,
                     fontweight=fontweight)

    plt.ylabel('True', fontsize=16)
    plt.xlabel('Predict', fontsize=16)

    plt.tight_layout()

    if max([len(label) for label in labels]) > 10:
        plt.subplots_adjust(bottom=0.2, left=0.2)

    if save_filename is not None:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')

    return plt


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.get_cmap("Blues"), labels=[], save_filename=None):
    plt.figure(figsize=(4, 3), dpi=600)
    plt.imshow(cm * 100, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90, size=12)
    plt.yticks(tick_marks, labels, size=12)
    for i in range(len(tick_marks)):
        for j in range(len(tick_marks)):
            if i != j:
                text = plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=10)
            elif i == j:
                if int(np.around(cm[i, j] * 100)) == 100:
                    text = plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=7,
                                    color='darkorange')
                else:
                    text = plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=10,
                                    color='darkorange')

    plt.tight_layout()
    if save_filename is not None:
        plt.savefig(save_filename, dpi=600, bbox_inches='tight')
    plt.close()


def calculate_confusion_matrix(Y, Y_hat, classes):
    n_classes = len(classes)
    conf = np.zeros([n_classes, n_classes])
    confnorm = np.zeros([n_classes, n_classes])

    for k in range(0, Y.shape[0]):
        i = list(Y[k, :]).index(1)
        j = int(np.argmax(Y_hat[k, :]))
        conf[i, j] = conf[i, j] + 1

    for i in range(0, n_classes):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

    right = np.sum(np.diag(conf))
    wrong = np.sum(conf) - right
    return confnorm, right, wrong


def l2_normalize(x: torch.Tensor, mode: str = 'full') -> torch.Tensor:
    if mode == 'full':
        return F.normalize(x, p=2, dim=(1, 2, 3), eps=1e-8)
    elif mode == 'channel':
        return F.normalize(x, p=2, dim=(1, 2), eps=1e-8)
    elif mode == 'spatial':
        return F.normalize(x, p=2, dim=1, eps=1e-8)
    else:
        return x
