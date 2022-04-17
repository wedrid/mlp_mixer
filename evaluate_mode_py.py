from mlp_mixer import * 
import json

####### EVAL PARAMS
one_batch = False
mod = 'mlp' #vit resnet
batchsize = 512
warmup = False


if mod == 'mlp':
    path = "./for_time_evaluation/"
    with open(path+'edo_out_hyper.json') as json_file:
        params = json.load(json_file)

    print(params)

    image_width_height = params['image_width_and_height'] #da cambiare a seconda della dimensione dell'immagine
    patch_dims = params['patch_width_and_height']
    # variable_name = value #paper value
    n_channels = params['hidden_dim_size (n_channels)'] #10 #512
    num_layers = params['number_of_layers'] #3
    mlp_dc_dimension = params['mlp_dc_dimension'] #8 #2048 # dc è la dimensione del channel mixing (l'ultimo mlp)
    mlp_ds_dimension = params['mlp_ds_dimension'] #8 #256 # ds è la dimensione del token mixing (il primo)

    model = MLP_mixer(img_h_w=image_width_height, patch_dim=patch_dims, n_channels=n_channels, num_mixers_layers=num_layers,
        hidden_dim_mlp_token=mlp_ds_dimension, hidden_dim_mlp_channel=mlp_dc_dimension, n_classes=10)

    #model.load_state_dict(torch.load(path+"final.pth"))
    #model.load_state_dict(torch.load(path+"edo_model_weights.pth"))
    model.eval()
elif mod == 'vit':
    import torch
    import json
    from vit import *
    print("LOADING VIT")

    # rand aug and scheduling learning rate applied

    # weights_path = './fine_tuning_vit_cifar100'
    weights_path = './for_time_evaluation'  # se tutto è nella stessa cartella (vit.py, hyperparams_ft.json, weights_71.pth, e questo file)

    print(weights_path)

    with open(weights_path + '/chiara_hyper.json') as json_file:
        hyper_params = json.load(json_file)

    # Definizione modello
    model = ViT(img_size=32, embed_dim=hyper_params['embed_dim'], num_channels=3,
                num_heads=hyper_params['num_heads'], num_layers=hyper_params['num_layers'],
                num_classes=10, patch_size=hyper_params['patch_size'],
                hidden_dim=hyper_params['hidden_dim'], dropout_value=hyper_params['dropout_value'])

    # print(model)

    model.eval()

    model.to(device)
elif mod == 'resnet':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.init as init

    from torch.autograd import Variable

    __all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

    def _weights_init(m):
        classname = m.__class__.__name__
        #print(classname)
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)

    class LambdaLayer(nn.Module):
        def __init__(self, lambd):
            super(LambdaLayer, self).__init__()
            self.lambd = lambd

        def forward(self, x):
            return self.lambd(x)


    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1, option='A'):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                if option == 'A':
                    """
                    For CIFAR10 ResNet paper uses option A.
                    """
                    self.shortcut = LambdaLayer(lambda x:
                                                F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
                elif option == 'B':
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(self.expansion * planes)
                    )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super(ResNet, self).__init__()
            self.in_planes = 16

            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
            self.linear = nn.Linear(64, num_classes)

            self.apply(_weights_init)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion

            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.avg_pool2d(out, out.size()[3])
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out




    def resnet110(num_cls):
        return ResNet(BasicBlock, [18, 18, 18], num_classes=num_cls)

    model = resnet110(num_cls=10)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #if device == 'cpu':
    #    model.load_state_dict(torch.load('./for_time_evaluation/resnet.pth',
    #                                     map_location='cpu'))
    #else:
    #    model.load_state_dict(torch.load('./for_time_evaluation/resnet.pth'))
    print(model)
else: 
    print("ERRORE, modello non correttamente specificato")

import torchvision.transforms as transforms
from get_dataloaders import * 


root = './cifar100_data' #if not in lab

dataloader_params = {'rand_augm_magnitude': 0, 'rand_augm_numops': 0, 'batch_size':batchsize}
one_batch = False
if one_batch: 
    dataloader_params['batch_size'] = 10000
else:
    dataloader_params['batch_size'] = dataloader_params['batch_size']


_, test_loader, _ = getCIFAR100Loaders(dataloader_params)

from tqdm.notebook import tqdm
import time
import numpy as np

loss_func = loss_func = nn.CrossEntropyLoss()

#processor warmup

if warmup:
    print("Processor warmup..")
    for _ in enumerate(tqdm(test_loader)): #numero esempi/batchsize TODO check
        images = images.to(device)
        labels = labels.to(device)
        for _ in range(5):
            predicted = model(images)
    print("..warmed up!")

if device == 'cuda':
    torch.cuda.synchronize()

times = list()
#sm = nn.Softmax(dim = 1)

with torch.no_grad():
        model.eval()
        val_accuracy = 0
        val_top_5_acc = 0
        temp = 0
        for i, (images, labels) in enumerate(tqdm(test_loader)): #numero esempi/batchsize TODO check
            images = images.to(device)
            labels = labels.to(device)
            
            # forward pass
            start = time.time()
            predicted = model(images)
            end = time.time()
            elapsed = end - start

            times.append(elapsed)
            #val_accuracy += get_accuracy(predicted, labels)
            #val_top_5_acc += get_top_5_accuracy(predicted, labels)
        #print(f"Lenght val loader: {len(val_loader)}, counter: {temp}")

        if one_batch:
            print(f"Elapsed: {elapsed}")
        times = np.array(times)
        mean_time = times.mean()
        print(f"Average time = {times.mean()}")