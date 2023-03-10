import numpy as np
from torch_geometric.data import DataLoader
import torch
from tqdm import trange
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import config_scheduling
import config_picking_success

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with torch.no_grad():
   torch.cuda.empty_cache()


# Choose goal

goal = 'scheduling'  # alternative: 'picking_success'
if goal == 'scheduling':
    cfg = config_scheduling
elif goal == 'picking_success':
    cfg = config_picking_success

# Tuned Parameters

learningRate = cfg.LR
hiddenLayers = cfg.HL
batchSize = cfg.BATCHSIZE
weightDecay = cfg.WEIGHTDECAY

# Set Seed

SeedNum = cfg.SEEDNUM
TorchSeed = cfg.TORCHSEED

np.random.seed(SeedNum)
torch.manual_seed(TorchSeed)
torch.cuda.manual_seed(TorchSeed)

# Load Dataset

print("Loading data_scripts...")
train_path = 'dataset/{}/data_train/'.format(goal)
train_dataset = cfg.DATASET(train_path)
val_path = 'dataset/{}/data_val/'.format(goal)
val_dataset = cfg.DATASET(val_path)
test_path = 'dataset/{}/data_test/'.format(goal)
test_dataset = cfg.DATASET(test_path)

train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
print("Done!")


# Initialize model, optimizer and loss

model = cfg.MODEL
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
scheduler = ReduceLROnPlateau(optimizer)
criterion = torch.nn.BCELoss()
batchAccuracy = cfg.ACCURACY()


# Parameters for plots

y_loss = {}
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def train_one_epoch():
    model.train()
    running_loss = 0.0
    running_corrects = 0.0
    real_scheduling = np.zeros(17)
    prob1 = 0
    prob2 = 0
    tot_nodes = 0.0
    step = 0

    all_y_pred = []
    all_y_true = []
    for i, batch in enumerate(train_loader, 0):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        batch.to(device)
        outputs = model(batch)
        loss = criterion(outputs, batch.y)
        # backward
        loss.backward()

        # optimize
        optimizer.step()

        # statistics
        # print(f'\nTrain Loss: {loss.item():.4f}, \t at iteration: {int(i):.4f}')
        running_loss += loss.item()
        running_corrects += batchAccuracy(outputs, batch.y, batch.batch)
        if goal == 'scheduling':
            for j in range(len(batch.y)):
                if outputs[j] > 0.5:
                    real_scheduling[batch.label[j] - 1] += 1
                    if batch.label[j] == 1:
                        prob1 += outputs[j]
                    elif batch.label[j] == 2:
                        prob2 += outputs[j]
        tot_nodes += len(batch.batch)
        step += 1

        # for additional metrics
        for j in range(len(outputs)):
            if outputs[j] > 0.5:
                outputs[j] = 1
            else:
                outputs[j] = 0
        y = torch.zeros_like(batch.y)
        for j in range(len(batch.y)):
            if batch.y[j] > 0.5:
                y[j] = 1
        all_y_pred.append(outputs.cpu().detach().numpy())
        all_y_true.append(y.cpu().detach().numpy())

    all_preds = np.concatenate(all_y_pred)
    all_labels = np.concatenate(all_y_true)
    calculate_metrics(all_preds, all_labels)

    running_loss = running_loss / step
    running_corrects = running_corrects / tot_nodes
    # for loss and accuracy plot
    y_loss['train'].append(running_loss)
    y_err['train'].append(running_corrects)
    if goal == 'scheduling':
        print('True scheduling of predicted as first (TRAIN): ', real_scheduling)
        print(f'\nAverage probability (TRAIN) of first being first: {float(prob1 / real_scheduling[0]):.4f},'
              f'\t or second: {float(prob2 / real_scheduling[1]):.4f}')
    return running_loss, running_corrects


def validation():
    model.eval()
    running_vloss = 0.0
    running_vcorrects = 0.0
    real_vscheduling = np.zeros(17)
    vprob1 = 0
    vprob2 = 0
    tot_vnodes = 0.0
    step = 0

    for i, vbatch in enumerate(val_loader):
        vbatch.to(device)
        voutputs = model(vbatch)
        vloss = criterion(voutputs, vbatch.y)
        running_vloss += vloss.item()
        running_vcorrects += batchAccuracy(voutputs, vbatch.y, vbatch.batch)
        if goal == 'scheduling':
            for j in range(len(vbatch.y)):
                if voutputs[j] > 0.5:
                    real_vscheduling[vbatch.label[j] - 1] += 1
                if vbatch.label[j] == 1:
                    vprob1 += voutputs[j]
                elif vbatch.label[j] == 2:
                    vprob2 += voutputs[j]
        tot_vnodes += len(vbatch.batch)
        step += 1

    avg_vloss = running_vloss / step
    avg_vcorrects = running_vcorrects / tot_vnodes

    y_loss['val'].append(avg_vloss)
    y_err['val'].append(avg_vcorrects)
    if goal == 'scheduling':
        print('True scheduling of predicted as first (VAL): ', real_vscheduling)
        print(f'\nAverage probability (VAL) of first being first: {float(vprob1 / real_vscheduling[0]):.4f},'
              f'\t or second: {float( vprob2 / real_vscheduling[1]):.4f}')
    return avg_vloss, avg_vcorrects


def test():
    model.eval()
    all_y_pred = []
    all_y_true = []
    real_tscheduling = np.zeros(17)
    tprob1 = 0
    tprob2 = 0

    for i, tbatch in enumerate(test_loader):
        tbatch.to(device)
        pred = model(tbatch)
        if goal == 'scheduling':
            for j in range(len(tbatch.y)):
                if pred[j] > 0.5:
                    real_tscheduling[tbatch.label[j] - 1] += 1
                if tbatch.label[j] == 1:
                    tprob1 += pred[j]
                elif tbatch.label[j] == 2:
                    tprob2 += pred[j]

        for j in range(len(pred)):
            if pred[j] > 0.5:
                pred[j] = 1
            else:
                pred[j] = 0
        y = torch.zeros_like(tbatch.y)
        for j in range(len(tbatch.y)):
            if tbatch.y[j] > 0.5:
                y[j] = 1
        all_y_pred.append(pred.cpu().detach().numpy())
        all_y_true.append(y.cpu().detach().numpy())

    all_preds = np.concatenate(all_y_pred)
    all_labels = np.concatenate(all_y_true)
    calculate_metrics(all_preds, all_labels)
    if goal == 'scheduling':
        print('True scheduling of predicted as first (TEST): ', real_tscheduling)
        print(f'\nAverage probability (TEST) of first being first: {float(tprob1 / real_tscheduling[0]):.4f},'
              f'\t or second: {float(tprob2 / real_tscheduling[1]):.4f}')


def calculate_metrics(y_pred, y_true):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_true, y_pred)}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average=None)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Precision: {precision_score(y_true, y_pred, average=None)}")
    print(f"Recall: {recall_score(y_true, y_pred, average=None)}")
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred))
    disp.plot()
    if BestModel:
        plt.savefig(os.path.join('./plots/plots_{}'.format(goal), 'CM{}_{}_{}_L2{}_{}.jpg'.format(Phase, hiddenLayers, batchSize, weightDecay, SeedNum)))


def draw_curve(current_epoch, cfg):
    if current_epoch != lastEpoch:
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train', linewidth=1)
        ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val', linewidth=1)
        ax1.plot(x_epoch, y_err['train'], 'bo-', label='train', linewidth=1)
        ax1.plot(x_epoch, y_err['val'], 'ro-', label='val', linewidth=1)
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    elif current_epoch == lastEpoch:
        ax0.text(0.5, 0.5, 'T' + str(best_loss),
                 horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes)
        ax0.text(0.5, 0.2, 'V' + str(best_vloss),
                 horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes)
        ax1.text(0.5, 0.5, 'T' + str(best_accuracy),
                 horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.text(0.5, 0.2, 'V' + str(best_vaccuracy),
                 horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.savefig(os.path.join('./plots/plots_{}'.format(goal), 'train_{}_{}_L2{}_{}.jpg'.format(cfg.HL, cfg.BATCHSIZE, cfg.WEIGHTDECAY, cfg.SEEDNUM)))


# Main

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
best_vloss = 1_000_000.
best_loss = 1_000_000.
best_vaccuracy = 0
best_accuracy = 0
early_stopping_counter = 0
NumEpochs = 300
lastEpoch = NumEpochs + 1
lastEpochlist = []
Phase = 'train'
BestModel = False
for epoch in trange(1, NumEpochs + 1):
    if early_stopping_counter <= 10:
        # Training
        train_loss, train_accuracy = train_one_epoch()
        # Validation
        val_loss, val_accuracy = validation()

        print(f'\nTrain Loss: {train_loss:.4f}, \tValidation Loss: {val_loss:.4f}')  # , \tTest Loss: {test_loss:.4f}
        print(f'\nTrain Accuracy: {train_accuracy:.4f}, \tValidation Accuracy: {val_accuracy:.4f}')

        scheduler.step(val_loss)

        # draw curve
        draw_curve(epoch, cfg)

        # Log the running loss averaged per batch for training, and validation
        writer.add_scalars('Training vs. Validation Loss', {'Training': train_loss, 'Validation': val_loss}, epoch + 1)
        writer.flush()

        # Track the best performance, and save the model's state
        if val_loss < best_vloss:
            best_vloss = val_loss
            model_path = 'best_models/best_models_{}/model_{}'.format(goal, timestamp)
            torch.save(model.state_dict(), model_path)
            early_stopping_counter = 0
            BestModel = True
        else:
            early_stopping_counter += 1
            BestModel = False

        if train_loss < best_loss:
            best_loss = train_loss

        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy

        if val_accuracy > best_vaccuracy:
            best_vaccuracy = val_accuracy
    else:
        lastEpochlist.append(epoch)
        continue

if early_stopping_counter >= 10:
    print("Early stopping due to no improvement.")

# Test
Phase = 'test'
BestModel = True  # to plot the confusion matrix of the loss
model.load_state_dict(torch.load(model_path))
test()

# to print loss and accuracy best values
if len(lastEpochlist) > 0:
    lastEpoch = int(lastEpochlist[0])
draw_curve(lastEpoch, cfg)

# Clear cuda cache
with torch.no_grad():
    torch.cuda.empty_cache()
