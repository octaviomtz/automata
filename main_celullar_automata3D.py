#!/usr/bin/env python
# coding: utf-8
import random
# from cellular_automaton import *
import itertools
import numpy as np
from scipy import ndimage
import os
import scipy
from copy import copy
from copy import deepcopy
from skimage.measure import block_reduce
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import imageio

from utils_cellular_automata import *
from scipy.stats import bernoulli
from scipy import stats
from scipy import signal 

path_img_dest = 'gifs/gifs_numpy/'

path_data = '/data/OMM/Datasets/LIDC_other_formats/LUNA_inpainted_cubes_for_GAN_v2/'
files = os.listdir(f'{path_data}original/')
files = np.sort(files)
block_name = files[116]

survive_single = list(range(27))
_ = [survive_single.remove(i) for i in [4,6,8,10]]
birth_single = [np.random.randint(0,8) for i in range(4)]
iter_total = 10


path_gifs1 = 'gifs/images_before_gifs/'
path_gifs2 = 'gifs/images_before_gifs2/'
path_gifs3 = 'gifs/images_before_gifs3/'
path_gifs4 = 'gifs/images_before_gifs4/'
grid_patches = []
cells_actives = []
grid_filtereds = []

last, orig, mask, mask_lungs = get_raw_nodule(path_data, block_name)
grid_patch, cells_active = add_seed_3D(last)
mask = ndimage.binary_dilation(mask, iterations=2)

for i in tqdm(range(iter_total)):
    grid_patches.append(grid_patch)
    cells_actives.append(cells_active)
    
    grid_prev = copy(grid_patch)
#     cells_active = ndimage.binary_dilation(cells_active, iterations=2)
    grid_neigh, grid_means = count_neighbors_and_get_means_3D_mask(grid_patch, mask, threshold=.2)
    grid_patch = survive_and_birth_individual_list_3D_mask(grid_neigh, grid_means, mask, grid_patch, survive_single, 
                                                         birth_single, cells_active)
    
    grid_filtered = filter_nodule_generated3_3D(grid_patch, cells_active)
    cells_active, grid_change_dilated = update_grid_cells_active(cells_active, grid_prev, grid_patch)
    grid_filtereds.append(grid_filtered)
    plot_for_gif_classes(grid_filtered[31], iter_total, i, (0,1))
    plot_for_gif_classes(cells_active[31], iter_total, i, (0,1), fold=path_gifs2)
    plot_for_gif_classes(grid_patch[31], iter_total, i, (0,1), fold=path_gifs3)
    
    # dilate to prevent having a fixed suare-like limit
    # mask = ndimage.binary_dilation(mask)
    
text_survive, text_birth = format_survive_and_birth_texts(survive_single, birth_single)
rule_name = f'v1x_{text_survive}_{text_birth}'
images = make_gif(path_gifs1)
imageio.mimsave(f'{path_img_dest}{rule_name}.gif', images)
images_cells_active = make_gif(path_gifs2)
imageio.mimsave(f'{path_img_dest}{rule_name}_active.gif', images_cells_active)
images_patch = make_gif(path_gifs3)
imageio.mimsave(f'{path_img_dest}{rule_name}_raw.gif', images_patch)