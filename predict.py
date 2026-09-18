import os
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

import tools


def predict(model, device_ids, weight_filepath, device, test_loader, classes, snrs, lbl, X_test, Y_test, test_idx, fig_base_dir,
            confusion_matrix_filepath, acc_mod_snr_filepath, f1_mod_snr_filepath, confusion_matrix_snr_path, acc_filepath, big_plot=False):
    model.load_state_dict(torch.load(weight_filepath, map_location=device))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    model.eval()

    acc = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    f1_mod_snr = np.zeros((len(classes), len(snrs)))
    i = 0
    correct = 0
    total = 0

    all_test_Y_hat = []
    all_Y_test = []

    with torch.no_grad():
        test_loader_with_progress = tqdm(test_loader, desc='测试进度',
                                         unit='batch', total=len(test_loader), ncols=100)

        for X_test_batch, Y_test_batch in test_loader_with_progress:
            X_test_batch, Y_test_batch = X_test_batch.to(device), Y_test_batch.to(device)

            output_test = model(X_test_batch)

            _, predicted = torch.max(output_test.data, 1)
            _, Y_test_batch_indices = torch.max(Y_test_batch, 1)
            total += Y_test_batch.size(0)
            correct += (predicted == Y_test_batch_indices).sum().item()

            all_test_Y_hat.extend(output_test.cpu().numpy())
            all_Y_test.extend(Y_test_batch.cpu().numpy())

            current_accuracy = 100 * correct / total
            test_loader_with_progress.set_postfix({
                'Accuracy': f'{current_accuracy:.2f}%',
                'Correct': f'{correct}/{total}'
            })

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')

    all_test_Y_hat = np.array(all_test_Y_hat)
    all_Y_test = np.array(all_Y_test)

    confnorm, _, _ = tools.calculate_confusion_matrix(all_Y_test, all_test_Y_hat, classes)
    confusion_df = pd.DataFrame(confnorm * 100, index=classes, columns=classes)
    confusion_df.to_csv(confusion_matrix_filepath, float_format='%.2f')
    if big_plot:
        tools.plot_confusion_matrix_big(confnorm, labels=classes,
                                        save_filename=os.path.join(fig_base_dir, 'total_confusion.png'))
    else:
        tools.plot_confusion_matrix(confnorm, labels=classes,
                                    save_filename=os.path.join(fig_base_dir, 'total_confusion.png'))

    tp = np.diag(confnorm)
    fp = np.sum(confnorm, axis=0) - tp
    fn = np.sum(confnorm, axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)

    f1_scores_per_class = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float),
                                    where=(precision + recall) != 0)
    f1_mod_snr = f1_scores_per_class

    print(f'计算整体混淆矩阵并绘制')

    for snr in snrs:
        test_SNRs = [lbl[x][1] for x in test_idx]
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        if isinstance(test_X_i, torch.Tensor):
            test_X_i_tensor = test_X_i
        else:
            test_X_i_tensor = torch.from_numpy(test_X_i)

        if isinstance(test_Y_i, torch.Tensor):
            test_Y_i_numpy = test_Y_i.cpu().numpy()
        else:
            test_Y_i_numpy = test_Y_i

        model = model.to(device)
        model.eval()

        batch_size = 128
        dataset = torch.utils.data.TensorDataset(test_X_i_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        all_outputs = []
        with torch.no_grad():
            for batch in dataloader:
                batch_x = batch[0].to(device)
                batch_output = model(batch_x)
                all_outputs.append(batch_output.cpu())

        test_Y_i_hat = torch.cat(all_outputs, dim=0).numpy()

        confnorm_i, cor, ncor = tools.calculate_confusion_matrix(test_Y_i_numpy, test_Y_i_hat, classes)
        acc[snr] = 1.0 * cor / (cor + ncor)

        with open(acc_filepath, 'a', newline='') as f0:
            writer = csv.writer(f0)
            writer.writerow([acc[snr]])

        confusion_df = pd.DataFrame(confnorm_i * 100, index=classes, columns=classes)
        confusion_matrix_each_snr_path = os.path.join(confusion_matrix_snr_path, f'confusion_matrix_{snr}.csv')
        confusion_df.to_csv(confusion_matrix_each_snr_path, float_format='%.2f')

        if big_plot:
            tools.plot_confusion_matrix_big(confnorm_i, labels=classes, title=f"Confusion Matrix (SNR={snr})",
                                            save_filename=os.path.join(fig_base_dir, f'Confusion(SNR={snr}).png'))
        else:
            tools.plot_confusion_matrix(confnorm_i, labels=classes, title=f"Confusion Matrix (SNR={snr})",
                                        save_filename=os.path.join(fig_base_dir, f'Confusion(SNR={snr}).png'))

        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i) / np.sum(confnorm_i, axis=1), 3)
        i += 1

        print(f'已完成:{snr}')

    overall_avg_f1 = np.mean(f1_mod_snr)
    print(f"Average F1 scores:{overall_avg_f1 * 100:.2f}%")

    dis_num = len(classes)
    for g in range(int(np.ceil(acc_mod_snr.shape[0] / dis_num))):
        beg_index = g * dis_num
        end_index = min((g + 1) * dis_num, acc_mod_snr.shape[0])

        plt.figure(figsize=(12, 10))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("Classification Accuracy for Each Modulation")

        for j in range(beg_index, end_index):
            plt.plot(snrs, acc_mod_snr[j], label=classes[j])
            for x, y in zip(snrs, acc_mod_snr[j]):
                plt.text(x, y, y, ha='center', va='bottom', fontsize=8)

        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(fig_base_dir, f'acc_with_mod_{g + 1}.png'))
        plt.close()

    df = pd.DataFrame(
        acc_mod_snr,
        index=classes,
        columns=snrs
    )
    df.to_csv(acc_mod_snr_filepath, float_format='%.4f')

    df_f1 = pd.DataFrame(
        f1_mod_snr,
        index=classes
    )
    df_f1.to_csv(f1_mod_snr_filepath, float_format='%.4f')

    plt.plot(snrs, [acc[x] for x in snrs])
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on RadioML 2016.10 Alpha")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_base_dir, 'each_acc.png'))
