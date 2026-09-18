import os
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from model.AVDSDN import AVDSDN
from data.Dataprocess_VMD import load_data_vmd
import tools
from train import Train
from predict import predict

train_enabled = False
eval_enabled = True

device_ids = [0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(device))
else:
    print(device)

data_filepath = '/data/zhengyurui/data/rmldata/rml16a/RML2016.10a_vmd_float32.pkl'

run_time = datetime.now().strftime('%Y%m%d_%H%M%S')

weight_dir = os.path.join('weights/rml16a', run_time)
training_loss_dir = os.path.join('training_loss', run_time)
acc_dir = os.path.join('acc', run_time)
fig_base_dir = os.path.join('figure', run_time)

if train_enabled:
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(training_loss_dir, exist_ok=True)

if eval_enabled:
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(fig_base_dir, exist_ok=True)

train_weight_filepath = os.path.join(weight_dir, 'weights_rml16.pth')
eval_weight_filepath = 'weights/rml16a/weights_rml16.pth'
csv_filepath = os.path.join(training_loss_dir, 'training_loss.csv')

acc_filepath = os.path.join(acc_dir, 'acc_rml16a.csv')
acc_mod_snr_filepath = os.path.join(acc_dir, 'acc_for_mod_rml16a.csv')
f1_mod_snr_filepath = os.path.join(acc_dir, 'f1_for_mod_rml16a.csv')
confusion_matrix_filepath = os.path.join(acc_dir, 'confusion_matrix_all.csv')
confusion_matrix_snr_path = acc_dir

print(f'Run time: {run_time}')
print(f'Train enabled: {train_enabled}, Eval enabled: {eval_enabled}')
print(f'Train weight path: {train_weight_filepath}')
print(f'Eval weight path: {eval_weight_filepath}')

nb_epoch = 200
batch_size = 256
learning_rate = 0.005
stop_num = 100
seed = 3409
torch.manual_seed(seed)
big_plot = False

(mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (
    train_idx, val_idx, test_idx) = load_data_vmd(seed=seed, filename=data_filepath)

X_train = tools.l2_normalize(torch.from_numpy(X_train), mode='full')
Y_train = torch.from_numpy(Y_train)
X_val = tools.l2_normalize(torch.from_numpy(X_val), mode='full')
Y_val = torch.from_numpy(Y_val)
X_test = tools.l2_normalize(torch.from_numpy(X_test), mode='full')
Y_test = torch.from_numpy(Y_test)

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
test_dataset = TensorDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

in_shp = list(X_train.shape[1:])
print("------------------------------------------------------------")
print(f"Train Data Shape: {X_train.shape}, Input Shape: {in_shp}")
print(f"Classes: {mods}")
print(f"Batch_size: {batch_size}")

net = AVDSDN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=6, min_lr=0.000001)

if train_enabled:
    Train(net, criterion, optimizer, scheduler, train_weight_filepath,
          csv_filepath, device, nb_epoch, train_loader, val_loader, stop_num)

if eval_enabled:
    predict(net, device_ids, eval_weight_filepath, device, test_loader, mods, snrs, lbl, X_test, Y_test, test_idx, fig_base_dir,
            confusion_matrix_filepath, acc_mod_snr_filepath, f1_mod_snr_filepath, confusion_matrix_snr_path, acc_filepath,
            big_plot=big_plot)
