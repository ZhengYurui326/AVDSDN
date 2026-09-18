import torch
import csv
from tqdm import tqdm


def Train(net, criterion, optimizer, scheduler, weight_filepath, csv_filepath, device, nb_epoch, train_loader, val_loader, stop_num):
    best_val_loss = float('inf')
    with open(csv_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Validation Loss', 'Validation Accuracy', 'Learning Rate'])

        for epoch in range(nb_epoch):
            net.train()
            total_loss = 0.0

            train_progress = tqdm(train_loader,
                                  desc=f'Epoch {epoch + 1}/{nb_epoch}',
                                  leave=True,
                                  ncols=100)

            for batch_idx, (X_train_batch, Y_train_batch) in enumerate(train_progress):
                X_train_batch, Y_train_batch = X_train_batch.to(device), Y_train_batch.to(device)

                optimizer.zero_grad()
                output = net(X_train_batch)
                loss = criterion(output, Y_train_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                train_progress.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': optimizer.param_groups[0]['lr']
                })

            net.eval()
            loss_val = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for X_val_batch, Y_val_batch in val_loader:
                    X_val_batch, Y_val_batch = X_val_batch.to(device), Y_val_batch.to(device)
                    output_val = net(X_val_batch)
                    loss_val += criterion(output_val, Y_val_batch).item()
                    _, predicted = torch.max(output_val.data, 1)
                    _, labels = torch.max(Y_val_batch, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            loss_val /= len(val_loader)
            accuracy = 100 * correct / total

            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(loss_val)
            writer.writerow([epoch + 1, loss_val, accuracy, current_lr])
            print(f"Epoch {epoch + 1}/{nb_epoch} | "
                  f"Val Loss: {loss_val:.4f} | "
                  f"Val Acc: {accuracy:.2f}% | "
                  f"LR: {current_lr:.6f} | "
                  f"Best: {best_val_loss:.4f}")

            if loss_val < best_val_loss:
                best_val_loss = loss_val
                last_improvement = epoch
                torch.save(net.state_dict(), weight_filepath)
            elif epoch - last_improvement > stop_num:
                print("Stop Training")
                break
