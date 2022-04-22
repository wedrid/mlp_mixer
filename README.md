# An MLP-mixer [1] implementation

The architecture is implemented in the file mlp_mixer.py
The file train.py allows for training on cifar10 and cifar100, compiled the json containing the hyperparameters. The training procedure includes various regularizations.
The implementation lacks of Dropout layers. 
The file pretrain.py allows for pretraining of the network, originally the small networks that were used (circa 1.1M parameters) was pretrained on downsampled imagenet 1k. 
For ImageNet the network is not hyperparametrized, but the idea was to do pretraining to let the network learn features to then be used for the downstream tasks of classification for CIFAR10 and CIFAR100. 

This repo is half of a work done in collaboration with Chiara Albisani, that aimed at comparing ViT [2] and MLP-mixer in our smaller scale; the work was done for a university assignment of machine learning. 
Her ViT implementation can be found here: https://github.com/chiaraalbi46/ViT 

Our results are synthetized in this presentation: https://docs.google.com/presentation/d/1gkpbro_fxi9ntAtzNo821gi6YvPTknwzx-MfhhPbvbE/edit?usp=sharing 

If the training code doesn't work, its most likely due to my removal of the logging object that logged parameters on comet.ml. The file containing the implementation should likely work. 

[1] Ilya Tolstikhin et al., 2021, MLP-Mixer: An all-MLP Architecture for Vision.
[2] Alexey Dosovitskiy et al., 2020, An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
