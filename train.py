from comet_ml import Experiment
import torch
import torchvision
import torchvision.transforms as transforms
from utils import * 
import time
from mlp_mixer import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import json
import matplotlib.pyplot as plt
from get_dataloaders import * 

#controllare che len(val) è len(train)
log = True
scheduling = False


def train(in_hyperparams, train_loader, val_loader, pretrained_model_path=None):
    #hyperparameters
    patch_dims = in_hyperparams['patch_width_and_height']
    # variable_name = value #paper value
    n_channels = in_hyperparams['hidden_dim_size (n_channels)'] #128 #256 #100 #512 #embed dim
    loss_func = nn.CrossEntropyLoss()
    learning_rate = in_hyperparams['learning_rate']
    weight_decay = in_hyperparams['weight_decay']
    num_layers = in_hyperparams['number_of_layers'] #8
    mlp_dc_dimension = in_hyperparams['mlp_dc_dimension'] #512 #1024 #2048 # dc è la dimensione del channel mixing (l'ultimo mlp)
    mlp_ds_dimension = in_hyperparams['mlp_ds_dimension'] #64 #128 #256 # ds è la dimensione del token mixing (il primo)
    mixup_alpha = in_hyperparams['mixup_alpha']
    num_epochs = in_hyperparams['epochs']
    batch_size = in_hyperparams['batch_size']

    examples = iter(train_loader)
    samples, labels = examples.next()
    print(samples.shape, labels.shape)

    img_sample = samples[0]
    print("MIAO")
    print(img_sample.shape)
    print(img_sample.shape)
    #plt.imshow(img_sample.permute(1, 2, 0))
    #plt.show()

    if log: 
        experiment = Experiment(
        api_key="xX6qWBFbiOreu0W3IrO14b9nB",
        project_name="mlp-mixer-final-trials",
        workspace="wedrid",
        )
        #experiment = Experiment(
        #api_key="xX6qWBFbiOreu0W3IrO14b9nB",
        #project_name="mlp-mixer",
        #workspace="wedrid",)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_width_height = img_sample.shape[1]

    model = MLP_mixer(img_h_w=image_width_height, patch_dim=patch_dims, n_channels=n_channels, n_classes=in_hyperparams['num_classes'], num_mixers_layers=num_layers,
        hidden_dim_mlp_token=mlp_ds_dimension, hidden_dim_mlp_channel=mlp_dc_dimension) #in this case 2 patches 16x16
    

    if pretrained_model_path is not None and False:
        model.load_state_dict(torch.load(pretrained_model_path))
    
    if weight_decay > 0: 
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay) #weight decay 0.1?
        print(f"WEIGHT DECAY IS: {weight_decay}")
    else: 
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    steps_total = len(train_loader)

    #ATTENZIONE: CAMBIARE IPERPARAMETRI ***PRIMAAAA*** DEL DICT SUCCESSIVO
    if scheduling: 
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=in_hyperparams['learning_rate'], pct_start=0.05,  # 0.05
                                                        total_steps=len(train_loader) * num_epochs,
                                                        anneal_strategy='cos')

    out_hyperparams = {
        "scheduling": scheduling,
        "dataset": in_hyperparams['dataset'],
        "num_classes": in_hyperparams['num_classes'],
        "rand_augm_numops": in_hyperparams['rand_augm_numops'],
        "rand_augm_magnitude": in_hyperparams['rand_augm_magnitude'],
        "comment": 'trial con parametri come fine tune',
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
    print(f"MIXUP ALPHA IS: {mixup_alpha}")
    if log: 
        experiment.log_parameters(out_hyperparams)
    model_path = generate_folder()
    with open(model_path+"/out_hyperparams.json", "w") as file:
        json.dump(out_hyperparams, file, indent=4)

    model.to(device)
    # training loop
    print("starting training loop")
    for epoch in tqdm(range(num_epochs)):
        start = time.time()
        model.train()
        train_accuracy = 0
        loss_ = 0
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            # [100, 3, 36, 36] is what is returned by iterator
            if mixup_alpha > 0:         
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha, device=device)
                #images, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
                predicted = model(images)
                loss = mixup_criterion(loss_func, predicted, labels_a, labels_b, lam)
                loss_ += loss.item()
                #_, predicted = torch.max(outputs.data, 1)
                train_accuracy += ((predicted.cpu().argmax(dim=-1) == labels).float().mean()).item()
            else:
                images = images.to(device)
                labels = labels.to(device)
                # forward pass
                predicted = model(images)
                loss = loss_func(predicted, labels)
                train_accuracy += ((predicted.argmax(dim=-1) == labels).float().mean()).item()
            #train_accuracy1 += get_accuracy(predicted, labels)
            #print(f"\nACCURACY {(train_accuracy)}")
            #print(f"prev acuracy {train_accuracy1}")
            #train_accuracy += get_accuracy(predicted, labels)

            # backwards pass
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

        # validation
        if scheduling:
            scheduler.step()
        with torch.no_grad():
            model.eval()
            val_accuracy = 0
            for i, (images, labels) in enumerate(tqdm(val_loader)): #numero esempi/batchsize TODO check
                # [100, 3, 36, 36] is what is returned by iterator

                images = images.to(device)
                labels = labels.to(device)
                predicted = model(images)
                loss = loss_func(predicted, labels)
                # forward pass
                

                #val_accuracy1 += get_accuracy(predicted, labels)
                val_accuracy += ((predicted.argmax(dim=-1) == labels).float().mean()).item()
                #print(f"\n val accuracy {val_accuracy}, previous: {val_accuracy1}")
            #print(f"Lenght val loader: {len(val_loader)}, counter: {temp}")
            val_accuracy /= len(val_loader) 
            if log: 
                experiment.log_metric("val epoch loss", loss.item(), step=epoch)
                experiment.log_metric("mean val epoch accuracy", val_accuracy, step=epoch)
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), model_path + f"checkpoint_epch_{epoch}.pth")
    torch.save(model.state_dict(), model_path + f"final.pth") 


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("-p", "--pretrained", dest="pretrained_model_path", default="",
                    help="pretrained model path, if empty does normal training" )

    args = parser.parse_args()

    with open('set_hyper_params.json') as json_file:
        in_hyperparams = json.load(json_file)
    if in_hyperparams['dataset'] == "CIFAR100":
        train_loader, val_loader, in_hyperparams['num_classes'] = getCIFAR100Loaders(in_hyperparams)
    elif in_hyperparams['dataset'] == "CIFAR10":
        train_loader, val_loader, in_hyperparams['num_classes'] = getCIFAR10Loaders(in_hyperparams)
        

    train(in_hyperparams, train_loader, val_loader)
