import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import torch
import pickle
import numpy as np
from torchvision.utils import save_image

from argparse import ArgumentParser

# usage example: 
# python imagenet_unpack.py -i "../datasets/imagenet/val_data" -o "../datasets/imagenet/validation/"

parser = ArgumentParser()
parser.add_argument("-i", "--infile", dest="in_file",
                    help="pickle file ( pickle )", metavar="FILE")
parser.add_argument("-o", "--outroot", dest="out_file", 
                    help="out root directory")
parser.add_argument("-p", "--prefix", dest="prefix", 
                    help="out root directory")

args = parser.parse_args()
print(args.in_file)
print(args.out_file)


with open(args.in_file, 'rb') as f:
    data = pickle.load(f)

tensors = data['data']
labels = data['labels']

root = args.out_file
Path(root).mkdir(parents=True, exist_ok=True)

#create class folder
for i in range(1, 1001):
    Path(root + str(i)).mkdir(parents=True, exist_ok=True)

for i in tqdm(range(len(labels))):
    np_tensor = torch.Tensor(np.array(tensors[i]).reshape((3,64,64)))/255
    save_image(np_tensor, root + "/" + str(labels[i]) + "/" + args.prefix + str(i) + '.png')
    #plt.imsave(root + "/" + str(labels[i]) + "/"+ str(i) + '.png', np_tensor)
    #print(labels[i])
