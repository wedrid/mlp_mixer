from comet_ml import Experiment
import json
from mlp_mixer import * 
import torch
from get_dataloaders import *
from tqdm import tqdm
import time
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

scheduling = False

path = "./models/20220401-122435/"
with open(path+'out_hyperparams.json') as json_file:
    params = json.load(json_file)

model = MLP_mixer(img_h_w=params['image_width_and_height'], patch_dim=params['patch_width_and_height'], n_channels=params['hidden_dim_size (n_channels)'], num_mixers_layers=params['number_of_layers'],
    hidden_dim_mlp_token=params['mlp_ds_dimension'], hidden_dim_mlp_channel=params['mlp_dc_dimension'], n_classes=1000)

if device == 'cpu':
    model.load_state_dict(torch.load(path+"checkpoint_epch_300.pth", map_location='cpu'))
else:
    model.load_state_dict(torch.load(path+"checkpoint_epch_300.pth"))
model.eval()

params['batch_size'] = 512
train_loader, val_loader, num_classes = getUpsampledCIFAR100Loaders(params)
num_in_features = model.fc_head.in_features

#cifar has 100 classes
model.fc_head = nn.Linear(num_in_features, num_classes)
model.to(device)

#define hyperparameters for the fine tuning
new_params = {'learning_rate': 0.0001, 'weight_decay': 1e-3, 'gradient_clipping': 1, 'comment': 'on cifar100', 'lr_sched': scheduling}

experiment = None

log = True
num_epochs = 500
optimizer = torch.optim.Adam(model.parameters(), lr = new_params['learning_rate'], weight_decay=new_params['weight_decay'])
datetime = time.strftime("%Y%m%d-%H%M%S")
loss_func = nn.CrossEntropyLoss()

experiment.log_parameters(new_params)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"Num parameters {params}")

if scheduling: 
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=new_params['learning_rate'], pct_start=0.05,  # 0.05
                                                        total_steps=len(train_loader) * num_epochs,
                                                        anneal_strategy='cos')

model.to(device)
# training loop
for epoch in tqdm(range(num_epochs)):
    start = time.time()
    model.train()
    train_accuracy = 0
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        # [100, 3, 36, 36] is what is returned by iterator
        images = images.to(device)
        labels = labels.to(device)
        
        # forward pass
        predicted = model(images)
        loss = loss_func(predicted, labels)
        train_accuracy += ((predicted.argmax(dim=-1) == labels).float().mean()).item()

        # backwards pass
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_value_(model.parameters(), new_params['gradient_clipping'])
        optimizer.step()
        
        if False and (i+1) % 100:
            print(f'epoch: {epoch+1} of {num_epochs}, step {i+1} of {steps_total}, loss = {loss.item():.4f}')
    print(f"Loss of epoch {epoch+1}: {loss.item():.4f}")
    train_accuracy /= len(train_loader)
    #print(f"TRAIN LOADER LENGTH: {len(train_loader)}")
    end = time.time()
    elapsed = end - start
    if log: 
        experiment.log_metric("train epoch loss", loss.item(), step=epoch)
        experiment.log_metric("mean train epoch accuracy", train_accuracy, step=epoch)
        experiment.log_metric("epoch time", elapsed, step = epoch)

    if scheduling:
        scheduler.step()

    # validation
    with torch.no_grad():
        model.eval()
        val_accuracy = 0
        temp = 0
        for i, (images, labels) in enumerate(tqdm(val_loader)): #numero esempi/batchsize TODO check
            # [100, 3, 36, 36] is what is returned by iterator
            images = images.to(device)
            labels = labels.to(device)
            
            # forward pass
            predicted = model(images)
            loss = loss_func(predicted, labels)
            val_accuracy += ((predicted.argmax(dim=-1) == labels).float().mean()).item()
        #print(f"Lenght val loader: {len(val_loader)}, counter: {temp}")
        val_accuracy /= len(val_loader) 
        if log: 
            experiment.log_metric("val epoch loss", loss.item(), step=epoch)
            experiment.log_metric("mean val epoch accuracy", val_accuracy, step=epoch)
    
    if epoch % 5 == 0:
        torch.save(model.state_dict(), path + f"finetune_checkpoint_epch_{epoch}_{datetime}.pth")

with open(path+f"hyper_{datetime}.json", "w") as file:
        json.dump(new_params, file, indent=4)