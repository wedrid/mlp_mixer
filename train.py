from comet_ml import Experiment
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from mlp_mixer import *
#from tqdm.notebook import tqdm
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import json
import time
from utils import * 

train_portion = 0.7

#controllare che len(val) è len(train)

with open('set_hyper_params.json') as json_file:
    params = json.load(json_file)

#hyperparameters
patch_dims = params['patch_width_and_height']
# variable_name = value #paper value
n_channels = params['hidden_dim_size (n_channels)'] #128 #256 #100 #512 #embed dim
loss_func = nn.CrossEntropyLoss()
learning_rate = params['learning_rate']
weight_decay = params['weight_decay']
num_layers = params['number_of_layers'] #8
mlp_dc_dimension = params['mlp_dc_dimension'] #512 #1024 #2048 # dc è la dimensione del channel mixing (l'ultimo mlp)
mlp_ds_dimension = params['mlp_ds_dimension'] #64 #128 #256 # ds è la dimensione del token mixing (il primo)
mixup_alpha = params['mixup_alpha']
num_epochs = params['epochs']
batch_size = params['batch_size']
randAugm_numops = params['rand_augm_numops']
randAugm_magn = params['rand_augm_magnitude']

if mixup_alpha < 0: 
    mixup = False
else:
    mixup = True


#pad_totensor_transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()]) # does the padding, images 32x32 become 36x36 (symmetric increase) so that are divisible by three and patches are 12x12
#pad_totensor_transform = transforms.Compose([transforms.ToTensor()]) #no pad, no normalization

my_transforms = transforms.Compose([ 
                            transforms.Resize((224,224)),
                            transforms.RandAugment(num_ops = randAugm_numops,magnitude = randAugm_magn ),
                            transforms.ToTensor(), #nota importante, ToTensor dev'essere sempre come ultima trasformazione
                            ])

#root = './imagenet' #if not in lab
root = '../datasets/cifar100'


dataset = torchvision.datasets.CIFAR100(root=root, train=True, transform = my_transforms, download=True)
train_subset, val_subset = torch.utils.data.random_split(dataset, [int(train_portion*len(dataset)), len(dataset) - int(train_portion*len(dataset))], generator=torch.Generator().manual_seed(1))
test_dataset = torchvision.datasets.CIFAR100(root=root, train=False, transform = transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(dataset=train_subset, shuffle=True, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(dataset=val_subset, shuffle=False, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(f"BATCH SIZE: {batch_size}")
print(f"Tran subset len: {len(train_subset)}")
print(f"Tran loader len: {len(train_loader)}")
print(f"Test: {len(train_subset)/batch_size}")

print(f"Val subset len: {len(val_subset)}")
print(f"Val subset len: {len(val_loader)}")
print(f"Test: {len(val_subset)/batch_size}")


print(f"Test subset len: {len(test_dataset)}")
print(f"Test subset len: {len(test_loader)}")
print(f"Test: {len(test_dataset)/batch_size}")

####


experiment = Experiment(
    api_key="xX6qWBFbiOreu0W3IrO14b9nB",
    project_name="mlp-mixer",
    workspace="wedrid",
)
if mixup: 
    experiment.add_tag('mixup')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

examples = iter(train_loader)
samples, labels = examples.next()

img_sample = samples[0]

image_width_height = img_sample.shape[1]

model = MLP_mixer(img_h_w=image_width_height, patch_dim=patch_dims, n_channels=n_channels, num_mixers_layers=num_layers,
    hidden_dim_mlp_token=mlp_ds_dimension, hidden_dim_mlp_channel=mlp_dc_dimension) #in this case 2 patches 16x16
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay) 

steps_total = len(train_loader)

#ATTENZIONE: CAMBIARE IPERPARAMETRI ***PRIMAAAA*** DEL DICT SUCCESSIVO

hyper_params = {
    "dataset": root,
    "rand_augm_numops": randAugm_numops,
    "rand_augm_magnitude": randAugm_magn,
    "comment": 'added weight decay',
    "train_size": len(train_loader),
    "validation_size": len(val_loader),
    "learning_rate": learning_rate,
    "epochs": num_epochs,
    "steps": steps_total,
    "batch_size": batch_size,
    "mixup_alpha": mixup_alpha, 
    "weight_decay": weight_decay,
    "image_width_and_height": image_width_height,
    "patch_width_and_height": patch_dims,
    "hidden_dim_size (n_channels)": n_channels,
    "number_of_layers": num_layers,
    "mlp_dc_dimension": mlp_dc_dimension,
    "mlp_ds_dimension": mlp_ds_dimension
}

experiment.log_parameters(hyper_params)

model_path = generate_folder()
with open(model_path+"/params.json", "w") as file:
    json.dump(hyper_params, file, indent=4)

model.to(device)
# training loop
for epoch in tqdm(range(num_epochs)):
    start = time.time()
    model.train()
    train_accuracy = 0
    #for i, (images, labels) in enumerate(tqdm(train_loader)):
    for i, (images, labels) in enumerate(train_loader):
        # [100, 3, 36, 36] is what is returned by iterator
        images = images.to(device)
        labels = labels.to(device)
        if not mixup: #without mixup regularization
            # forward pass
            predicted = model(images)
            loss = loss_func(predicted, labels)
            
        else:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha, device=device)
            #images, labels_a, labels_b = map(Variable, ) no because Variable is deprecated
            predicted = model(images)
            loss = mixup_criterion(loss_func, predicted, labels_a, labels_b, lam)
            
        train_accuracy += get_accuracy(predicted, labels)
        # backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if False and (i+1) % 100:
            print(f'epoch: {epoch+1} of {num_epochs}, step {i+1} of {steps_total}, loss = {loss.item():.4f}')
    print(f"Loss of epoch {epoch+1}: {loss.item():.4f}")
    train_accuracy /= len(train_loader)
    #print(f"TRAIN LOADER LENGTH: {len(train_loader)}")
    end = time.time()
    elapsed = end - start
    experiment.log_metric("train epoch loss", loss.item(), step = epoch)
    experiment.log_metric("mean train epoch accuracy", train_accuracy, step = epoch)
    experiment.log_metric("epoch time", elapsed, step = epoch)
    # validation
    with torch.no_grad():
        model.eval()
        val_accuracy = 0
        temp = 0
        for i, (images, labels) in enumerate((val_loader)): #numero esempi/batchsize TODO check
            # [100, 3, 36, 36] is what is returned by iterator
            images = images.to(device)
            labels = labels.to(device)
            
            # forward pass
            predicted = model(images)
            loss = loss_func(predicted, labels)
            val_accuracy += get_accuracy(predicted, labels)
        #print(f"Lenght val loader: {len(val_loader)}, counter: {temp}")
        val_accuracy /= len(val_loader) 
        experiment.log_metric("val epoch loss", loss.item(), step=epoch) #TODO average loss?
        experiment.log_metric("mean val epoch accuracy", val_accuracy, step=epoch)
    
    if epoch % 10 == 0:
        torch.save(model.state_dict(), model_path + f"checkpoint_epch_{epoch}.pth")
torch.save(model.state_dict(), model_path + f"final.pth")
