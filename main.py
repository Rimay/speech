from torch.utils.data import DataLoader, random_split
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter   

import utils as u
from data_provider import dataset
from model import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('./log')

def train(model, train_dl, epochs):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, 
                                                    max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=epochs,
                                                    anneal_strategy='linear'
                                                    )
    cnt = 0                                        
    for e in range(epochs):
        losses = 0.0
        correct_pred, total_pred = 0, 0
        for i, data in enumerate(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = (inputs - inputs.mean()) / inputs.std()

            opt.zero_grad()

            pred = model(inputs)
            loss = crit(pred, labels)
            loss.backward()

            opt.step()
            scheduler.step()

            losses += loss.item()

            _, l = torch.max(pred, 1)
            correct_pred += (l == labels).sum().item()
            total_pred += l.shape[0]
            if (i + 1) % 10 == 0:
                print('[e {:}, itr {:}] loss {:.3f}'.format(e, i, loss))
                writer.add_scalar('loss', loss, cnt)
                writer.add_scalar('precision', (l == labels).sum().item() / l.shape[0], cnt)
                cnt += 1
                
        batch_num = len(train_dl)
        print('------------------------------------------------------------------')
        print('epoch: {:}, loss: {:.2f}, precision: {:.2f}'.format(e, losses / batch_num, correct_pred / total_pred))
        print('------------------------------------------------------------------')
        if e % 10 == 9:
            torch.save(model, './models/epoch_{:}.model'.format(e + 1))


def infer(model, val_dl):
    correct_pred, total_pred = 0, 0
    with torch.no_grad():
        for data in val_dl:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = (inputs - inputs.mean()) / inputs.std()

            pred = model(inputs)

            _, l = torch.max(pred, 1)
            correct_pred += (l == labels).sum().item()
            total_pred += l.shape[0]
    print('------------------------------------------------------------------')
    print('Val Accuracy: {:.2f}, Total items:{:}'.format(correct_pred / total_pred, total_pred))
    print('------------------------------------------------------------------')
    torch.save(model, './models/final.model')


if __name__ == '__main__':
    print('----- device type')
    print(device)

    print('----- loading dataset')
    ds = dataset()
    num_items = len(ds)
    num_train = int(num_items * 0.8)
    num_val = num_items - num_train
    print('----- train num: {:}'.format(num_train))
    print('----- val num: {:}'.format(num_val))

    train_ds, val_ds = random_split(ds, [num_train, num_val])
    train_dl, val_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True),\
                        torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

    print('----- creating model')
    model = model().to(device)


    print('----- start training')
    train(model, train_dl, epochs=1000)

    print('----- inferring')
    infer(model, val_dl)
