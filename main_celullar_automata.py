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
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import imageio

from utils_cellular_automata import *
from scipy.stats import bernoulli
from scipy import stats
from scipy import signal 

path_img_dest = 'gifs/gifs_numpy/'
path_data = '/data/OMM/Datasets/LIDC_other_formats/LUNA_inpainted_cubes_for_GAN_v2/'
survive = [1,2,4,5,6,7,8]
birth = [3]
files = os.listdir(f'{path_data}original/')
files = np.sort(files)
block_name = files[200]

mini_patched, mini_orig, mini_last, mini_mask, mini_mask_not_lungs, mini_patched_clean, mini_patched_borders, grid_cells_active = pick_a_raw_nodule_and_set_a_nodule_seed(path_data, block_name)

grid_patch = copy(mini_patched_clean)
iter_total = 20
grid_mses = []
grid_cells_actives = []
grid_patches = []
grid_filtereds = []
mini_mask_dilated_twice = ndimage.binary_dilation(mini_mask,iterations=2)

for i in tqdm_notebook(range(iter_total)):
    grid_prev = copy(grid_patch)
    grid_patches.append(grid_patch)
    grid_cells_actives.append(grid_cells_active)
    
    grid_neigh, grid_means = count_neighbors_and_get_means_mask(grid_patch, mini_mask_dilated_twice, .2) #or grid_cells_active instead of mini_mask_dilated_twice?
    grid_patch, text_survive, text_birth = survive_and_birth(grid_neigh, grid_means, grid_patch, survive, birth, mini_mask_dilated_twice) #or grid_cells_active instead of mini_mask_dilated_twice?
    grid_cells_active = update_grid_cells_active(grid_cells_active, grid_prev, grid_patch)
    grid_filtered = filter_nodule_generated2(grid_patch, mini_mask_not_lungs, grid_cells_active, mini_patched_borders)
    
    grid_diff = (mini_orig*mini_mask-grid_patch*mini_mask)**2
    grid_mse = np.mean(grid_diff)
    
    plot_for_gif_classes(grid_filtered, iter_total, i, (0,1))
    plot_for_gif_classes(grid_patch, iter_total, i, (0,1), 'gifs/images_before_gifs2/')
    plot_for_gif_classes(grid_cells_active, iter_total, i, (0,1), 'gifs/images_before_gifs3/')
    
    grid_filtereds.append(grid_filtered)
    grid_mses.append(grid_mse)
# ===
images = make_gif('gifs/images_before_gifs/')
images_patch = make_gif('gifs/images_before_gifs2/')

ca_seed = 0
rule_name = f'CAv1_{text_survive}_{text_birth}_seed_{ca_seed}'
imageio.mimsave(f'{path_img_dest}{rule_name}.gif', images)
imageio.mimsave(f'{path_img_dest}grid_{rule_name}.gif', images_patch)
plot_inpain_orig_seed_generated(f'{path_img_dest}{rule_name}.png', mini_last, mini_orig, mini_patched_clean, grid_filtered)

