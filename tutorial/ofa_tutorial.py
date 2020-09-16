# %%
import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import time
import random
import math
import copy
from matplotlib import pyplot as plt

from ofa.model_zoo import ofa_net
from ofa.utils import download_url

# from ofa.tutorial.accuracy_predictor import AccuracyPredictor
# from ofa.tutorial.flops_table import FLOPsTable
# from ofa.tutorial.latency_table import LatencyTable
# from ofa.tutorial.evolution_finder import EvolutionFinder
# from ofa.tutorial.imagenet_eval_helper import evaluate_ofa_subnet, evaluate_ofa_specialized
from ofa.tutorial import AccuracyPredictor, FLOPsTable, LatencyTable, EvolutionFinder
from ofa.tutorial import evaluate_ofa_subnet, evaluate_ofa_specialized

# set random seed
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)

# %% [markdown]
# Now it's time to determine which device to use for neural network inference in the rest of this tutorial. If your machine is equipped with GPU(s), we will use the GPU by default. Otherwise, we will use the CPU.

# %%
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)
    print('Using GPU.')
else:
    print('Using CPU.')

# %% [markdown]
# Good! Now you have successfully configured the environment! It's time to import the **OFA network** for the following experiments.
# The OFA network used in this tutorial is built upon MobileNetV3 with width multiplier 1.2, supporting elastic depth (2, 3, 4) per stage, elastic expand ratio (3, 4, 6), and elastic kernel size (3, 5 7) per block.

# %%
ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)
print('The OFA Network is ready.')

# %% [markdown]
# Now, let's build the ImageNet dataset and the corresponding dataloader. Notice that **if you're using the CPU,
# we will skip ImageNet evaluation by default** since it will be very slow.
# If you are using the GPU, in case you don't have the full dataset,
# we will download a subset of ImageNet which contains 2,000 images (~250M) for testing.
# If you do have the full ImageNet dataset on your machine, just specify it in `imagenet_data_path` and the downloading script will be skipped.

# %%
if cuda_available:
    # path to the ImageNet dataset
    print("Please input the path to the ImageNet dataset.\n")
    imagenet_data_path = './imagenet_1k' #input()

    # if 'imagenet_data_path' is empty, download a subset of ImageNet containing 2000 images (~250M) for test
    if not os.path.isdir(imagenet_data_path):
        os.makedirs(imagenet_data_path, exist_ok=True)
        download_url('https://hanlab.mit.edu/files/OnceForAll/ofa_cvpr_tutorial/imagenet_1k.zip', model_dir='data')
        os.system(' cd data && unzip imagenet_1k 1>/dev/null && cd ..')
        os.system(' cp -r data/imagenet_1k/* $imagenet_data_path')  #! pc: copy not done properly
        os.system(' rm -rf data')
        print('%s is empty. Download a subset of ImageNet for test.' % imagenet_data_path)

    print('The ImageNet dataset files are ready.')
else:
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')

# %% [markdown]
# Now you have configured the dataset. Let's build the dataloader for evaluation.
# Again, this will be skipped if you are in a CPU environment.

# %%
if cuda_available:
    # The following function build the data transforms for test
    def build_val_transform(size):
        return transforms.Compose([
            transforms.Resize(int(math.ceil(size / 0.875))),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root=os.path.join(imagenet_data_path, 'val'),
            transform=build_val_transform(224)
        ),
        batch_size=250,  # test batch size
        shuffle=True,
        num_workers=16,  # number of workers for the data loader
        pin_memory=True,
        drop_last=False,
    )
    print('The ImageNet dataloader is ready.')
else:
    data_loader = None
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')

# %% [markdown]
# ## 2. Using Pretrained Specialized OFA Sub-Networks
# ![](https://hanlab.mit.edu/files/OnceForAll/figures/select_subnets.png)
# The specialized OFA sub-networks are "small" networks sampled from the "big" OFA network as is indicated in the figure above.
# The OFA network supports over $10^{19}$ sub-networks simultaneously, so that the deployment cost for multiple scenarios can be saved by 16$\times$ to 1300$\times$ under 40 deployment scenarios.
# Now, let's play with some of the sub-networks through the following interactive command line prompt (**Notice that for CPU users, this will be skipped**).
# We recommend you to try a smaller sub-network (e.g., the sub-network for pixel1 with 20ms inference latency constraint) so that it takes less time to evaluate the model on ImageNet.

# %%
if cuda_available:
    net_id = evaluate_ofa_specialized(imagenet_data_path, data_loader)
    print('Finished evaluating the pretrained sub-network: %s!' % net_id)
else:
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')

# %% [markdown]
# ## 3 Efficient Deployment with OFA Networks
#  
# You have now successfully prepared the whole environment for the experiment!
# In the next step, we will introduce **how to get efficient, specialized neural networks within minutes**
# powered by the OFA network.
# 
# ### 3.1 Latency-Constrained Efficient Deployment on Samsung Note10
# 
# The key components of very fast neural network deployment are **accuracy predictors** and **efficiency predictors**.
# For the accuracy predictor, it predicts the Top-1 accuracy of a given sub-network on a **holdout validation set**
# (different from the official 50K validation set) so that we do **NOT** need to run very costly inference on ImageNet
# while searching for specialized models. Such an accuracy predictor is trained using an accuracy dataset built with the OFA network.
# 
# %%
# accuracy predictor
accuracy_predictor = AccuracyPredictor(
    pretrained=True,
    device='cuda:0' if cuda_available else 'cpu'
)

print('The accuracy predictor is ready!')
print(accuracy_predictor.model)

# %% [markdown]
# Now, we have the powerful **accuracy predictor**. We then introduce two types of **efficiency predictors**: the latency predictor and the FLOPs predictor. 
# 
# The intuition of having efficiency predictors, especially the latency predictor, is that measuring the latency of a sub-network on-the-fly is also costly, especially for mobile devices.
# The latency predictor is designed to eliminate this cost.
# Let's load a latency predictor we built beforehand for the Samsung Note10.

# %%
target_hardware = 'note10'
latency_table = LatencyTable(device=target_hardware)
print('The Latency lookup table on %s is ready!' % target_hardware)

# %% [markdown]
# So far, we have defined both the accuracy predictor and the latency predictor. Now, let's experience **very fast model specialization** on Samsung Note10 with these two powerful predictors! 
# 
# **Notice**: The predicted accuracy is on a holdout validation set of 10K images, not the official 50K validation set.
# But they are highly positive-correlated.

# %%
""" Hyper-parameters for the evolutionary search process
    You can modify these hyper-parameters to see how they influence the final ImageNet accuracy of the search sub-net.
"""
latency_constraint = 25  # ms, suggested range [15, 33] ms
P = 100  # The size of population in each generation
N = 500  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
params = {
    'constraint_type': target_hardware, # Let's do FLOPs-constrained search
    'efficiency_constraint': latency_constraint,
    'mutate_prob': 0.1, # The probability of mutation in evolutionary search
    'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
    'efficiency_predictor': latency_table, # To use a predefined efficiency predictor.
    'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
}

# build the evolution finder
finder = EvolutionFinder(**params)

# start searching
result_lis = []
st = time.time()
best_valids, best_info = finder.run_evolution_search()
result_lis.append(best_info)
ed = time.time()
print('Found best architecture on %s with latency <= %.2f ms in %.2f seconds! '
      'It achieves %.2f%s predicted accuracy with %.2f ms latency on %s.' %
      (target_hardware, latency_constraint, ed-st, best_info[0] * 100, '%', best_info[-1], target_hardware))

# visualize the architecture of the searched sub-net
_, net_config, latency = best_info
ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
print('Architecture of the searched sub-net:')
print(ofa_network.module_str)

# %% [markdown]
# Great! You get your specialized neural network with **just a few seconds**!
# You can go back to the last cell and modify the hyper-parameters to see how they affect the search time and the accuracy.
# 
# We also provided an interface below to draw a figure comparing your searched specialized network and other efficient neural networks such as MobileNetV3 and ProxylessNAS.
# 
# **Notice**: For ease of comparison, we recommend you to choose a latency constraint between 15ms and 33ms.

# %%
# evaluate the searched model on ImageNet
if cuda_available:
    top1s = []
    latency_list = []
    for result in result_lis:
        _, net_config, latency = result
        print('Evaluating the sub-network with latency = %.1f ms on %s' % (latency, target_hardware))
        top1 = evaluate_ofa_subnet(
            ofa_network,
            imagenet_data_path,
            net_config,
            data_loader,
            batch_size=250,
            device='cuda:0' if cuda_available else 'cpu')
        top1s.append(top1)
        latency_list.append(latency)

    plt.figure(figsize=(4,4))
    plt.plot(latency_list, top1s, 'x-', marker='*', color='darkred',  linewidth=2, markersize=8, label='OFA')
    plt.plot([26, 45], [74.6, 76.7], '--', marker='+', linewidth=2, markersize=8, label='ProxylessNAS')
    plt.plot([15.3, 22, 31], [73.3, 75.2, 76.6], '--', marker='>', linewidth=2, markersize=8, label='MobileNetV3')
    plt.xlabel('%s Latency (ms)' % target_hardware, size=12)
    plt.ylabel('ImageNet Top-1 Accuracy (%)', size=12)
    plt.legend(['OFA', 'ProxylessNAS', 'MobileNetV3'], loc='lower right')
    plt.grid(True)
    plt.show()
    print('Successfully draw the tradeoff curve!')
else:
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')

# %% [markdown]
# **Notice:** You can further significantly improve the accuracy of the searched sub-net by fine-tuning it on the ImageNet training set.
# Our results after fine-tuning for 25 epochs are as follows:
# ![](https://hanlab.mit.edu/files/OnceForAll/figures/diverse_hardware.png)
# 
# 
# ### 3.2 FLOPs-Constrained Efficient Deployment
# 
# Now, let's proceed to the final experiment of this tutorial: efficient deployment under FLOPs constraint. We use the same accuracy predictor since accuracy predictors are agnostic to the types of efficiency constraint (mobile latency / FLOPs). For the efficiency predictor, we change the latency lookup table to a flops lookup table. You can run the code below to setup it in a few seconds.

# %%
flops_lookup_table = FLOPsTable(
    device='cuda:0' if cuda_available else 'cpu',
    batch_size=1,
)
print('The FLOPs lookup table is ready!')

# %% [markdown]
# Now, you can start a FLOPs-constrained neural architecture search. Here, we directly generate **an entire tradeoff curve** for you. Please notice that the time it takes to get each data point will get longer and longer (but always less than 30 seconds) because smaller FLOPs-constraint is more difficult to meet.
# 
# If you are using CPUs, you will be able to see a "predicted holdout validation set accuracy - FLOPs" tradeoff curve, which can be obtained in just a  minute.
# 
# If you are using GPUs, besides the curve mentioned above, we will also evaluate all the models you designed on the ImageNet validation set (**Again, it will be better if you have the full ImageNet validation set**, but it's also OK if you downloaded the subset above) and generate an "ImageNet 50K validation set accuracy - FLOPs" tradeoff curve. We will also plot competing methods such as ProxylessNAS, MobileNetV3, and EfficientNet in this curve for your reference. The estimated time to get the two curves is less than 10 minutes.
# 
# Please notice that it usually takes ** hundreds/thousands of hours** to generate an accuracy-FLOPs tradeoff curve for ProxylessNAS / MobileNetV3 / EfficientNet, but generating the tradeoff curve for our OFA takes just a few minutes, as you will experience soon.

# %%
""" Hyper-parameters for the evolutionary search process
    You can modify these hyper-parameters to see how they influence the final ImageNet accuracy of the search sub-net.
"""
P = 100  # The size of population in each generation
N = 500  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
params = {
    'constraint_type': 'flops', # Let's do FLOPs-constrained search
    'efficiency_constraint': 600,  # FLops constraint (M), suggested range [150, 600]
    'mutate_prob': 0.1, # The probability of mutation in evolutionary search
    'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
    'efficiency_predictor': flops_lookup_table, # To use a predefined efficiency predictor.
    'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
}

# build the evolution finder
finder = EvolutionFinder(**params)

# start searching
result_lis = []
for flops in [600, 400, 350]:
    st = time.time()
    finder.set_efficiency_constraint(flops)
    best_valids, best_info = finder.run_evolution_search()
    ed = time.time()
    # print('Found best architecture at flops <= %.2f M in %.2f seconds! It achieves %.2f%s predicted accuracy with %.2f MFLOPs.' % (flops, ed-st, best_info[0] * 100, '%', best_info[-1]))
    result_lis.append(best_info)

plt.figure(figsize=(4,4))
plt.plot([x[-1] for x in result_lis], [x[0] * 100 for x in result_lis], 'x-', marker='*', color='darkred',  linewidth=2, markersize=8, label='OFA')
plt.xlabel('FLOPs (M)', size=12)
plt.ylabel('Predicted Holdout Top-1 Accuracy (%)', size=12)
plt.legend(['OFA'], loc='lower right')
plt.grid(True)
plt.show()

# %% [markdown]
# Let's evaluate the searched models on ImageNet if GPU is available:

# %%
if cuda_available:
    # test the searched model on the test dataset (ImageNet val)
    top1s = []
    flops_lis = []
    for result in result_lis:
        _, net_config, flops = result
        print('Evaluating the sub-network with FLOPs = %.1fM' % flops)
        top1 = evaluate_ofa_subnet(
            ofa_network,
            imagenet_data_path,
            net_config,
            data_loader,
            batch_size=250,
            device='cuda:0' if cuda_available else 'cpu')
        print('-' * 45)
        top1s.append(top1)
        flops_lis.append(flops)

    plt.figure(figsize=(8,4))
    plt.subplot(1, 2, 1)
    plt.plot([x[-1] for x in result_lis], [x[0] * 100 for x in result_lis], 'x-', marker='*', color='darkred',  linewidth=2, markersize=8, label='OFA')
    plt.xlabel('FLOPs (M)', size=12)
    plt.ylabel('Predicted Holdout Top-1 Accuracy (%)', size=12)
    plt.legend(['OFA'], loc='lower right')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(flops_lis, top1s, 'x-', marker='*', color='darkred',  linewidth=2, markersize=8, label='OFA')
    plt.plot([320, 581], [74.6, 76.7], '--', marker='+', linewidth=2, markersize=8, label='ProxylessNAS')
    plt.plot([219, 343], [75.2, 76.6], '--', marker='^', linewidth=2, markersize=8, label='MobileNetV3')
    plt.plot([390, 700], [76.3, 78.8], '--', marker='>', linewidth=2, markersize=8, label='EfficientNet')
    plt.xlabel('FLOPs (M)', size=12)
    plt.ylabel('ImageNet Top-1 Accuracy (%)', size=12)
    plt.legend(['OFA', 'ProxylessNAS', 'MobileNetV3', 'EfficientNet'], loc='lower right')
    plt.grid(True)
    plt.show()

# %% [markdown]
# **Notice:** Again, you can further improve the accuracy of the search sub-net by fine-tuning it on ImageNet.
# The final accuracy is much better than training the same architecture from scratch.
# Our results are as follows:
# ![](https://hanlab.mit.edu/files/OnceForAll/figures/imagenet_80_acc.png)
# ![](https://hanlab.mit.edu/files/OnceForAll/figures/cnn_imagenet_new.png)
# 
# Congratulations! You've finished all the content of this tutorial!
# Hope you enjoy playing with the OFA Networks. If you are interested,  please refer to our paper and GitHub Repo for further details.
# 
# ## Reference
# [1] CVPR'20 tutorial: **AutoML for TinyML with Once-for-All Network**. [[talk]](https://www.youtube.com/watch?v=fptQ_eJ3Uc0&feature=youtu.be).
# 
# [1] Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang and Song Han.
# **Once for All: Train One Network and Specialize It for Efficient Deployment**. In *ICLR* 2020.
# [[paper]](https://arxiv.org/abs/1908.09791), [[code]](https://github.com/mit-han-lab/once-for-all), [[talk]](https://www.youtube.com/watch?v=a_OeT8MXzWI).
# 
# [2] Han Cai, Ligeng Zhu and Song Han. **ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware**.
# In *ICLR* 2019. [[paper]](https://arxiv.org/abs/1812.00332), [[code]](https://github.com/MIT-HAN-LAB/ProxylessNAS).
# 

