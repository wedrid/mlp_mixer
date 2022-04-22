import torch
import torchvision
import torchvision.transforms as transforms


def getCIFAR100Loaders(in_params, root='./cifarCento_data'):
    randAugm_numops = in_params['rand_augm_numops']
    randAugm_magn = in_params['rand_augm_magnitude']
    pad_totensor_transform = transforms.Compose([
        transforms.RandAugment(num_ops = randAugm_numops,magnitude = randAugm_magn),
        transforms.ToTensor()]) #no pad, no normalization

    dataset = torchvision.datasets.CIFAR100(root=root, train=True, transform=pad_totensor_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root=root, train=False, transform=transforms.ToTensor())


    train_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=in_params['batch_size'])
    val_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=in_params['batch_size'], shuffle=False)

    print(f"BATCH SIZE: {in_params['batch_size']}")
    print(f"Tran subset len: {len(dataset)}")
    print(f"Tran loader len: {len(train_loader)}")
    print(f"Test: {len(dataset)/in_params['batch_size']}")

    print(f"Val/test subset len: {len(test_dataset)}")
    print(f"Val/test subset len: {len(val_loader)}")
    print(f"Val/Test: {len(test_dataset)/in_params['batch_size']}")

    print(f"Test subset len: {len(test_dataset)}")
    print(f"Test subset len: {len(val_loader)}")
    print(f"Test: {len(test_dataset)/in_params['batch_size']}")
    return train_loader, val_loader, len(dataset.classes)

def getCIFAR10Loaders(in_params, root='./cifar10_data'):
    randAugm_numops = in_params['rand_augm_numops']
    randAugm_magn = in_params['rand_augm_magnitude']
    pad_totensor_transform = transforms.Compose([
        transforms.RandAugment(num_ops = randAugm_numops,magnitude = randAugm_magn),
        transforms.ToTensor()]) #no pad, no normalization

    dataset = torchvision.datasets.CIFAR10(root=root, train=True, transform=pad_totensor_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, transform=transforms.ToTensor())
    print(f"LEN TEST DATASET: {len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=in_params['batch_size'])
    val_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=in_params['batch_size'], shuffle=False)

    print(f"BATCH SIZE: {in_params['batch_size']}")
    print(f"Tran subset len: {len(dataset)}")
    print(f"Tran loader len: {len(train_loader)}")
    print(f"Test: {len(dataset)/in_params['batch_size']}")

    print(f"Val/test subset len: {len(test_dataset)}")
    print(f"Val/test subset len: {len(val_loader)}")
    print(f"Val/Test: {len(test_dataset)/in_params['batch_size']}")

    print(f"Test subset len: {len(test_dataset)}")
    print(f"Test subset len: {len(val_loader)}")
    print(f"Test: {len(test_dataset)/in_params['batch_size']}")
    return train_loader, val_loader, len(dataset.classes)
    

def getImagenetLoaders(in_params, root='../datasets/imagenet'):
    randAugm_numops = in_params['rand_augm_numops']
    randAugm_magn = in_params['rand_augm_magnitude']
    my_transform = transforms.Compose([
        transforms.RandAugment(num_ops = randAugm_numops, magnitude = randAugm_magn),
        transforms.ToTensor()]) #no pad, no normalization

    train_dataset = torchvision.datasets.ImageFolder(root=root + "/train", transform=my_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=root + "/validation", transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=in_params['batch_size'])
    val_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=in_params['batch_size'], shuffle=False)

    print(f"BATCH SIZE: {in_params['batch_size']}")
    print(f"Tran subset len: {len(train_dataset)}")
    print(f"Tran loader len: {len(train_loader)}")
    print(f"Test: {len(train_dataset)/in_params['batch_size']}")

    print(f"Val/test subset len: {len(test_dataset)}")
    print(f"Val/test subset len: {len(val_loader)}")
    print(f"Val/Test: {len(test_dataset)/in_params['batch_size']}")

    print(f"Test subset len: {len(test_dataset)}")
    print(f"Test subset len: {len(val_loader)}")
    print(f"Test: {len(test_dataset)/in_params['batch_size']}")
    return train_loader, val_loader, len(train_dataset.classes)

def getUpsampledCIFAR100Loaders(in_params, root='./cifar100_data'):
    randAugm_numops = in_params['rand_augm_numops']
    randAugm_magn = in_params['rand_augm_magnitude']
    with_rand_augm_transform = transforms.Compose([
        transforms.RandAugment(num_ops = randAugm_numops,magnitude = randAugm_magn),
        transforms.Resize([64,64]),
        transforms.ToTensor()]) #no pad, no normalization
    no_rand_augm_transform = transforms.Compose([
        transforms.Resize([64,64]),
        transforms.ToTensor()])

    dataset = torchvision.datasets.CIFAR100(root=root, train=True, transform=with_rand_augm_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root=root, train=False, transform=no_rand_augm_transform)


    train_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=in_params['batch_size'])
    val_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=in_params['batch_size'], shuffle=False)

    print(f"BATCH SIZE: {in_params['batch_size']}")
    print(f"Tran subset len: {len(dataset)}")
    print(f"Tran loader len: {len(train_loader)}")
    print(f"Test: {len(dataset)/in_params['batch_size']}")

    print(f"Val/test subset len: {len(test_dataset)}")
    print(f"Val/test subset len: {len(val_loader)}")
    print(f"Val/Test: {len(test_dataset)/in_params['batch_size']}")

    print(f"Test subset len: {len(test_dataset)}")
    print(f"Test subset len: {len(val_loader)}")
    print(f"Test: {len(test_dataset)/in_params['batch_size']}")
    return train_loader, val_loader, len(dataset.classes)

def getUpsampledCIFAR10Loaders(in_params, root='./cifar100_data'):
    randAugm_numops = in_params['rand_augm_numops']
    randAugm_magn = in_params['rand_augm_magnitude']
    with_rand_augm_transform = transforms.Compose([
        transforms.RandAugment(num_ops = randAugm_numops,magnitude = randAugm_magn),
        transforms.Resize([64,64]),
        transforms.ToTensor()]) #no pad, no normalization
    no_rand_augm_transform = transforms.Compose([
        transforms.Resize([64,64]),
        transforms.ToTensor()])

    dataset = torchvision.datasets.CIFAR10(root=root, train=True, transform=with_rand_augm_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, transform=no_rand_augm_transform)


    train_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=in_params['batch_size'])
    val_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=in_params['batch_size'], shuffle=False)

    print(f"BATCH SIZE: {in_params['batch_size']}")
    print(f"Tran subset len: {len(dataset)}")
    print(f"Tran loader len: {len(train_loader)}")
    print(f"Test: {len(dataset)/in_params['batch_size']}")

    print(f"Val/test subset len: {len(test_dataset)}")
    print(f"Val/test subset len: {len(val_loader)}")
    print(f"Val/Test: {len(test_dataset)/in_params['batch_size']}")

    print(f"Test subset len: {len(test_dataset)}")
    print(f"Test subset len: {len(val_loader)}")
    print(f"Test: {len(test_dataset)/in_params['batch_size']}")
    return train_loader, val_loader, len(dataset.classes)