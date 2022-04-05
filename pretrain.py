from comet_ml import Experiment
import torch
import torchvision
import torchvision.transforms as transforms
from utils import * 
from get_dataloaders import * 
import time
from mlp_mixer import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import json
import matplotlib.pyplot as plt

log = True
#controllare che len(val) è len(train)


def train(in_hyperparams, train_loader, val_loader, model=None):
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
    print(img_sample.shape)
    print(img_sample.shape)
    #plt.imshow(img_sample.permute(1, 2, 0))
    #plt.show()

    if log: 
        experiment = Experiment(
            api_key="xX6qWBFbiOreu0W3IrO14b9nB",
            #project_name="mlp-mixer-pretraining",
            project_name="mlp-mixer",
            workspace="wedrid",
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_width_height = img_sample.shape[1]

    model = MLP_mixer(img_h_w=image_width_height, patch_dim=patch_dims, n_channels=n_channels, n_classes=in_hyperparams['num_classes'], num_mixers_layers=num_layers,
        hidden_dim_mlp_token=mlp_ds_dimension, hidden_dim_mlp_channel=mlp_dc_dimension) #in this case 2 patches 16x16
    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=in_hyperparams['weight_decay']) #weight decay 0.1?
    print(f"WEIGHT DECAY: {in_hyperparams['weight_decay']}")
    steps_total = len(train_loader)

    #ATTENZIONE: CAMBIARE IPERPARAMETRI ***PRIMAAAA*** DEL DICT SUCCESSIVO

    out_hyperparams = {
        "dataset": "-",
        "rand_augm_numops": in_hyperparams['rand_augm_numops'],
        "rand_augm_magnitude": in_hyperparams['rand_augm_magnitude'],
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

    experiment.log_parameters(out_hyperparams)
    model_path = generate_folder()
    with open(model_path+"/out_hyperparams.json", "w") as file:
        json.dump(out_hyperparams, file, indent=4)

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
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), model_path + f"checkpoint_epch_{epoch}.pth")
    torch.save(model.state_dict(), model_path + f"final.pth") 


if __name__ == "__main__":
    with open('set_hyper_params.json') as json_file:
        in_hyperparams = json.load(json_file)
    
    #train_loader, val_loader, num_classes = getImagenetLoaders(in_hyperparams)
    train_loader, val_loader, num_classes = getCIFAR100Loaders(in_hyperparams)
    in_hyperparams['num_classes'] = num_classes
    train(in_hyperparams, train_loader, val_loader)