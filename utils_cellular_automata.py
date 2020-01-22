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
from numba import jit

from scipy.stats import bernoulli
from scipy import stats
from scipy import signal 
import seaborn as sns

def plot_4_genetic_algorithm(auto_mask, auto_orig, auto_patched, figsize=(10,6)):
    fig, ax = plt.subplots(1,4,figsize=figsize)
    ax[0].imshow(auto_mask)
    ax[0].axis('off')
    ax[1].imshow(auto_orig, vmin=0, vmax=1)
    ax[1].axis('off')
    ax[2].imshow(auto_patched, vmin=0, vmax=1)
    ax[2].axis('off')
    ax[3].imshow(np.abs(auto_orig - auto_patched), vmin=0, vmax=1)
    ax[3].axis('off')

def plot_for_gif_classes(image_to_save,num_iter, i, col_range, fold = 'gifs/images_before_gifs/'):
    fig, ax = plt.subplots(1,2, gridspec_kw = {'width_ratios':[12, 1]}, figsize=(10,7))
    ax[0].imshow(image_to_save, vmin=col_range[0], vmax=col_range[1], cmap='viridis')
    ax[0].axis('off')
    ax[1].axvline(x=.5, c='k')
    if isinstance(i, int):
        ax[1].scatter(.5, i, c='k')
    else:
        ax[1].scatter(.5, 0, c='k')
    ax[1].set_ylim([num_iter, 0])
    ax[1].yaxis.tick_right()
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
    # ax[1].xticks([], [])
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["bottom"].set_visible(False)
    ax[1].spines["left"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    fig.tight_layout()
    plt.subplots_adjust(wspace=.04, hspace=0)
    if isinstance(i, int):
        plt.savefig(f'{fold}iter {i:05d}.jpeg',bbox_inches = 'tight',pad_inches = 0)
    else: # this part is to save the initial figures (they are placed first because they have less zeroes)
        i = int(i)
        plt.savefig(f'{fold}iter {0:0{i}d}.jpeg',bbox_inches = 'tight',pad_inches = 0)
    plt.close(fig)

def save_original(image_to_save, path_img_dest, num_iter, id_name, name_extension):
    name_extension = str(name_extension)
    fig, ax = plt.subplots(1,2, gridspec_kw = {'width_ratios':[12, 1]}, figsize=(10,7))
    ax[0].imshow(image_to_save, cmap='viridis', vmin=0, vmax=1)
    ax[0].axis('off')
    ax[1].axvline(x=.5, c='k')
    ax[1].set_ylim([num_iter, 0])
    ax[1].yaxis.tick_right()
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["bottom"].set_visible(False)
    ax[1].spines["left"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    plt.subplots_adjust(wspace=.04, hspace=0)
    fig.savefig(f'{path_img_dest}{id_name} {name_extension}.jpeg',
                bbox_inches = 'tight',pad_inches = 0)
    plt.close(fig)

def get_nodule(path_data, block_name):
    path_last = f'{path_data}inpainted/'
    path_orig = f'{path_data}original/'
    path_mask_lungs = f'{path_data}mask lungs/'
    path_mask = f'{path_data}mask/'
    
    last = np.load(f'{path_last}{block_name}')
    orig = np.load(f'{path_orig}{block_name}')
    mask = np.load(f'{path_mask}{block_name[:-1]}z')
    mask = mask.f.arr_0
    mask_lungs = np.load(f'{path_mask_lungs}{block_name[:-1]}z')
    mask_lungs = mask_lungs.f.arr_0
    
    return last, orig, mask, mask_lungs

def find_nodule_with_mask(last, orig, mask, mask_lungs):
    z,y,x = np.where(mask==1)
    zz = int(np.median(z))
    auto_last = last[zz]
    auto_orig = orig[zz]
    auto_mask = mask[zz]
    auto_mask_lungs = mask_lungs[zz]
    y,x = np.where(auto_mask==1)
    xx = int(np.median(x))
    yy = int(np.median(y))
    auto_patched = auto_mask*auto_last + (-auto_mask+1)*auto_orig
    return auto_last, auto_orig, auto_mask, auto_mask_lungs, auto_patched, yy, xx

def get_central_slice(last, orig, mask, mask_lungs):
    '''The function comes from find_nodule_with_mask. When we were analyzing LIDC
    in this functions is where we were patchin the original image with the inpainted
    nodule using the mask. Now there is no need for that because we already read
    inpainted inserted'''
    z,y,x = np.where(mask==1)
    zz = int(np.median(z))
    auto_last = last[zz]
    auto_orig = orig[zz]
    auto_mask = mask[zz]
    auto_mask_lungs = mask_lungs[zz]
    y,x = np.where(auto_mask==1)
    xx = int(np.median(x))
    yy = int(np.median(y))
    return auto_last, auto_orig, auto_mask, auto_mask_lungs, auto_last, yy, xx

def coords_limits_using_mask_lungs(auto_last, auto_orig, auto_mask, auto_mask_lungs, auto_patched):
    mask_lungs_min_y = np.min(np.where(auto_mask_lungs==1)[0])
    mask_lungs_max_y = np.max(np.where(auto_mask_lungs==1)[0])
    mask_lungs_min_x = np.min(np.where(auto_mask_lungs==1)[1])
    mask_lungs_max_x = np.max(np.where(auto_mask_lungs==1)[1])

    mini_mask = auto_mask[mask_lungs_min_y:mask_lungs_max_y,mask_lungs_min_x:mask_lungs_max_x]
    mini_mask_lungs = auto_mask_lungs[mask_lungs_min_y:mask_lungs_max_y,mask_lungs_min_x:mask_lungs_max_x]
    mini_orig = auto_orig[mask_lungs_min_y:mask_lungs_max_y,mask_lungs_min_x:mask_lungs_max_x]
    mini_last = auto_last[mask_lungs_min_y:mask_lungs_max_y,mask_lungs_min_x:mask_lungs_max_x]
    mini_patched = copy(auto_patched[mask_lungs_min_y:mask_lungs_max_y,mask_lungs_min_x:mask_lungs_max_x])
    return mini_orig, mini_mask, mini_mask_lungs, mini_patched, mask_lungs_min_y, mask_lungs_min_x, mini_last

def make_gif(path):
    files_for_gif = os.listdir(path)
    files_for_gif = np.sort(files_for_gif)
    images = []
    for filename in files_for_gif:
        filename = f'{path}{filename}'
        images.append(imageio.imread(filename))
        os.remove(filename)
    return images

def count_neighbors_and_get_means(grid, threshold = 0.5):
    grid_sums = copy(grid)
    grid_means = copy(grid)
    for i in range(1,np.shape(grid)[0]-1):
        for j in range(1,np.shape(grid)[1]-1):
            # count alive neighbors
            s = 0
            s += np.sum([grid[i-1,j-1] > threshold, grid[i,j-1] > threshold, grid[i+1,j-1] > threshold])
            s += np.sum([grid[i-1,j] > threshold, grid[i+1,j] > threshold])
            s += np.sum([grid[i-1,j+1] > threshold, grid[i,j+1] > threshold, grid[i+1,j+1] > threshold])
            grid_sums[i,j] = s
            # get mean neighbors
            m = 0
            m += np.sum([grid[i-1,j-1] , grid[i,j-1] , grid[i+1,j-1]])
            m += np.sum([grid[i-1,j] , grid[i+1,j] ])
            m += np.sum([grid[i-1,j+1] , grid[i,j+1] , grid[i+1,j+1]])
            m = m/8
            grid_means[i,j] = m
    return grid_sums, grid_means

def count_neighbors_and_get_means_mask(grid, mini_mask_dilated_twice, threshold = 0.5):
    '''count the number of active neighbors and their means, BUT, only of those
    cells that are inside the mini_mask_dilated_twice'''
    grid_sums = copy(grid)
    grid_means = copy(grid)
    y,x = np.where(mini_mask_dilated_twice==1)

    for i in range(1,np.shape(grid)[0]-1):
        if i in y:
            for j in range(1,np.shape(grid)[1]-1):
                if j in x:
                    # count alive neighbors
                    s = 0
                    s += np.sum([grid[i-1,j-1] > threshold, grid[i,j-1] > threshold, grid[i+1,j-1] > threshold])
                    s += np.sum([grid[i-1,j] > threshold, grid[i+1,j] > threshold])
                    s += np.sum([grid[i-1,j+1] > threshold, grid[i,j+1] > threshold, grid[i+1,j+1] > threshold])
                    grid_sums[i,j] = s
                    # get mean neighbors
                    m = 0
                    m += np.sum([grid[i-1,j-1] , grid[i,j-1] , grid[i+1,j-1]])
                    m += np.sum([grid[i-1,j] , grid[i+1,j] ])
                    m += np.sum([grid[i-1,j+1] , grid[i,j+1] , grid[i+1,j+1]])
                    m = m/8
                    grid_means[i,j] = m
    return grid_sums, grid_means

def get_new_birth(grid_diff):
    '''Get new birth weights from the squared difference loss.
    First filter the difference to get a smoother grid.
    Then normalize to 9 (no birth possible) and 1 (very fast growth)
    birth=[3] is optimal'''
    grid_gauss = ndimage.filters.gaussian_filter(grid_diff, 1)
    grid_weights = (8*grid_gauss/grid_gauss.max())
    grid_weights = 9.3-grid_weights
    grid_weights = ndimage.filters.gaussian_filter(grid_weights, 1)
    grid_weights = grid_weights.astype(int)
    grid_weights = np.expand_dims(grid_weights, -1)
    return grid_weights

def adjust_survive_birth(survive_or_birth, mini_patched_):
    '''copy the survive or birth to create one for each pixel. Example:
    a = adjust_survive_birth([3,4],np.zeros((2,2)))
    [[[3, 4],[3, 4]],
     [[3, 4],[3, 4]]]
    '''
    survs_or_births = np.repeat([survive_or_birth], np.shape(mini_patched_)[0] * np.shape(mini_patched_)[1], axis=0)
    survs_or_births = survs_or_births.reshape((np.shape(mini_patched_)[0] , np.shape(mini_patched_)[1], len(survive_or_birth)))
    return survs_or_births

def format_survive_and_birth_texts(survive_single_, birth_single_):
    survive_text = [str(i) for i in survive_single_]
    survive_text = ''.join(survive_text)
    birth_text = [str(i) for i in birth_single_]
    birth_text = ''.join(birth_text)
    return survive_text, birth_text

def survive_and_birth_individual(neighbors, means, values, survives, births):
    grid_new = copy(values)
    for i in range(1,np.shape(neighbors)[0]-1):
        for j in range(1,np.shape(neighbors)[1]-1):
            
            # SURVIVE
            if .3 < grid_new[i,j] < .6: 
                grid_new[i,j] = np.squeeze([values[i,j]+.1 if neighbors[i,j] in survives[i,j] else values[i,j]-.1])
            # BIRTH
            elif grid_new[i,j] < .1:
                rand_close_to_alive = random.gauss(0.55,.05)
                last_cell_state = grid_new[i,j]
                if neighbors[i,j] in births[i,j]:
                    grid_new[i,j] = np.mean((rand_close_to_alive, means[i,j], last_cell_state))
            # LIMIT
            elif grid_new[i,j] > .6:
                rand_close_to_alive = random.gauss(0.55,.05)
                grid_new[i,j] = rand_close_to_alive
            else: pass
    
    return grid_new

def filter_nodule_generated(grid1, mini_mask_not_lungs):
    '''Find the pixels where a nodule is generated. Remove the lung borders using dilation,
    then find the largest connected component. Filter the area of the largest component, 
    then place the filtered image back (remove the filtered borders [too dark]). Finally add the lung borders.'''
    # remove the lung borders. dilate and find the largest component
    grid_change = grid1 > .2
    grid_change = grid_change * (~mini_mask_not_lungs)
    grid_change = ndimage.morphology.binary_dilation(grid_change)
    grid_labeled, nr = ndimage.label(grid_change)
    sizes = ndimage.sum(grid_change, grid_labeled,range(1,nr+1)) 
    elem_max = (np.where(sizes==sizes.max())[0] + 1)[0]
    largest_comp = grid_labeled == elem_max
    # filter the area of the largest component,
    # then place the filtered image back (remove the filtered borders [too dark])
    y,x = np.where(largest_comp==1)
    y_max = np.max(y); y_min = np.min(y)
    x_max = np.max(x); x_min = np.min(x)
    grid_filter = grid1[y_min:y_max,x_min:x_max]
    grid_filter = signal.wiener(grid_filter)
    grid_filter_added = copy(grid1)
    grid_filter_added[y_min:y_max,x_min:x_max] = grid_filter
    #borders
    grid_filter_added[y_min:y_max,x_min-1:x_min+1] = grid1[y_min:y_max,x_min-1:x_min+1]
    grid_filter_added[y_min:y_max,x_max-1:x_max+1] = grid1[y_min:y_max,x_max-1:x_max+1]
    grid_filter_added[y_min-1:y_min+1,x_min:x_max] = grid1[y_min-1:y_min+1,x_min:x_max]
    grid_filter_added[y_max-1:y_max+1,x_min:x_max] = grid1[y_max-1:y_max+1,x_min:x_max]
    # Add lung borders
    grid_filter_added = grid_filter_added*(~mini_mask_not_lungs)+mini_patched_borders*mini_mask_not_lungs
    return grid_filter_added

def filter_nodule_generatedX(grid1, mini_mask_not_lungs, mini_patched_borders):
    '''Find the pixels where a nodule is generated. Remove the lung borders using dilation,
    then find the largest connected component. Filter the area of the largest component, 
    then place the filtered image back (remove the filtered borders [too dark]). Finally add the lung borders.'''
    # remove the lung borders. dilate and find the largest component
    grid_change = grid1 > .2
    grid_change = grid_change * (~mini_mask_not_lungs)
    grid_change = ndimage.morphology.binary_dilation(grid_change)
    grid_labeled, nr = ndimage.label(grid_change)
    sizes = ndimage.sum(grid_change, grid_labeled,range(1,nr+1)) 
    elem_max = (np.where(sizes==sizes.max())[0] + 1)[0]
    largest_comp = grid_labeled == elem_max
    # filter the area of the largest component,
    # then place the filtered image back (remove the filtered borders [too dark])
    y,x = np.where(largest_comp==1)
    y_max = np.max(y); y_min = np.min(y)
    x_max = np.max(x); x_min = np.min(x)
    grid_filter = grid1[y_min:y_max,x_min:x_max]
    grid_filter = signal.wiener(grid_filter)
    grid_filter_added = copy(grid1)
    grid_filter_added[y_min:y_max,x_min:x_max] = grid_filter
    #borders
    grid_filter_added[y_min:y_max,x_min-1:x_min+1] = grid1[y_min:y_max,x_min-1:x_min+1]
    grid_filter_added[y_min:y_max,x_max-1:x_max+1] = grid1[y_min:y_max,x_max-1:x_max+1]
    grid_filter_added[y_min-1:y_min+1,x_min:x_max] = grid1[y_min-1:y_min+1,x_min:x_max]
    grid_filter_added[y_max-1:y_max+1,x_min:x_max] = grid1[y_max-1:y_max+1,x_min:x_max]
    # Add lung borders
    grid_filter_added = grid_filter_added*(~mini_mask_not_lungs)+mini_patched_borders*mini_mask_not_lungs
    return grid_filter_added

def filter_nodule_generatedTEMP(grid1, mini_mask_not_lungs, mini_patched_borders):
    '''Find the pixels where a nodule is generated. Remove the lung borders using
    dilation, then find the largest connected component. Filter the area of the 
    largest component, then place the filtered image back (remove the filtered 
    borders [too dark]). Finally add the lung borders.'''
    # remove the lung borders. dilate and find the largest component
    grid_change = grid1 > .2
    grid_change = grid_change * (mini_mask_not_lungs)
    grid_change = ndimage.morphology.binary_dilation(grid_change)
    grid_labeled, nr = ndimage.label(grid_change)
    sizes = ndimage.sum(grid_change, grid_labeled,range(1,nr+1)) 
    elem_max = (np.where(sizes==sizes.max())[0] + 1)[0]
    largest_comp = grid_labeled == elem_max
    # filter the area of the largest component,
    # then place the filtered image back (remove the filtered borders [too dark])

    fig, ax = plt.subplots(1,3,figsize=(12,3))
    ax[0].imshow(grid1)
    ax[1].imshow(mini_mask_not_lungs)
    ax[2].imshow(mini_patched_borders)

    y,x = np.where(largest_comp==1)
    
    y_max = np.max(y); y_min = np.min(y)
    x_max = np.max(x); x_min = np.min(x)
    grid_filter = grid1[y_min:y_max,x_min:x_max]
    grid_filter = signal.wiener(grid_filter)
    grid_filter_added = copy(grid1)
    grid_filter_added[y_min:y_max,x_min:x_max] = grid_filter
    #borders
    grid_filter_added[y_min:y_max,x_min-1:x_min+1] = grid1[y_min:y_max,x_min-1:x_min+1]
    grid_filter_added[y_min:y_max,x_max-1:x_max+1] = grid1[y_min:y_max,x_max-1:x_max+1]
    grid_filter_added[y_min-1:y_min+1,x_min:x_max] = grid1[y_min-1:y_min+1,x_min:x_max]
    grid_filter_added[y_max-1:y_max+1,x_min:x_max] = grid1[y_max-1:y_max+1,x_min:x_max]
    # Add lung borders
    grid_filter_added = grid_filter_added*(~mini_mask_not_lungs)+mini_patched_borders*mini_mask_not_lungs
    return grid_filter_added

## NON-UNIFORM CA 3 FUNCTIONS

def get_new_birth_list(grid_diff):
    '''Get new birth weights from the squared difference loss.
    First filter the difference to get a smoother grid.
    Then normalize to 9 (no birth possible) and 1 (very fast growth)
    birth=[3] is optimal'''
    grid_gauss = ndimage.filters.gaussian_filter(grid_diff, 1)
    grid_weights = 9 - (8.3*grid_gauss/grid_gauss.max())
    grid_weights = ndimage.filters.gaussian_filter(grid_weights, 1)
    grid_weights = grid_weights.astype(int)
    grid_weights = np.expand_dims(grid_weights, -1)
    grid_weights = grid_weights.tolist()
    return grid_weights

def get_new_multiple_birth_list(grid_diff):
    '''Get new birth weights from the squared difference loss.
    First filter the difference to get a smoother grid.
    Then normalize to 9 (no birth possible) and 1 (very fast growth)
    birth=[3] is optimal'''
    grid_gauss = ndimage.filters.gaussian_filter(grid_diff, 1)
    grid_weights = 5 - (4.3*grid_gauss/grid_gauss.max())
    grid_weights = ndimage.filters.gaussian_filter(grid_weights, 1)
    grid_weights = grid_weights.astype(int)
    grid_weights = np.expand_dims(grid_weights, -1)
    grid_weights = grid_weights.tolist()
    ## multiple births
    grid_weights_multiple = deepcopy(grid_weights)
    y,x,_ = np.where(np.asarray(grid_weights) <= 3)
    for i,j in zip(y,x):
        grid_weights_multiple[i][j].append(np.random.randint(1,4))
    return grid_weights_multiple, grid_weights 

def initialize_multiple_survive_birth_list(survive_or_birth, mini_patched_):
    '''copy the survive or birth to create one for each pixel. Example:
    a = adjust_survive_birth([3,4],np.zeros((2,2)))
    [[[3, 4],[3, 4]],
     [[3, 4],[3, 4]]]
    '''
    survs_or_births = np.repeat([survive_or_birth], np.shape(mini_patched_)[0] * np.shape(mini_patched_)[1], axis=0)
    survs_or_births = survs_or_births.reshape((np.shape(mini_patched_)[0] , np.shape(mini_patched_)[1], len(survive_or_birth)))
    survs_or_births = survs_or_births.tolist()
    return survs_or_births

def update_grid_cells_active(grid_cells_active_, grid_prev_, grid_patch_):
    grid_cells_active_dilated = ndimage.binary_dilation(grid_cells_active_)
    grid_change = np.abs(grid_prev_ - grid_patch_) > 0
    grid_change_dilated = ndimage.binary_dilation(grid_change)
    grid_cells_active = ((grid_change_dilated & grid_cells_active_dilated) + grid_cells_active_) >0
#     grid_cells_active_border = np.abs(grid_cells_active - grid_cells_active_dilated) >0
    return grid_cells_active, grid_change_dilated#, grid_cells_active_border

def count_neighbors_and_get_means2(grid, threshold = 0.5):
    '''Count and return the:
    -Amount of neighbors alive
    -Mean value of the neighbors
    -Mean value of the neighbors and their neighbors'''
    grid_sums = copy(grid)
    grid_means = copy(grid)
    grid_means2 = copy(grid)
    for i in range(1,np.shape(grid)[0]-1):
        for j in range(1,np.shape(grid)[1]-1):
            #print(i,j)
            # count alive neighbors
            s = 0
            s += np.sum([grid[i-1,j-1] > threshold, grid[i,j-1] > threshold, grid[i+1,j-1] > threshold])
            s += np.sum([grid[i-1,j] > threshold, grid[i+1,j] > threshold])
            s += np.sum([grid[i-1,j+1] > threshold, grid[i,j+1] > threshold, grid[i+1,j+1] > threshold])
            grid_sums[i,j] = s
            # get mean neighbors
            m = 0
            m += np.sum([grid[i-1,j-1] , grid[i,j-1] , grid[i+1,j-1]])
            m += np.sum([grid[i-1,j] , grid[i+1,j] ])
            m += np.sum([grid[i-1,j+1] , grid[i,j+1] , grid[i+1,j+1]])
            grid_means[i,j] = m / 8
            # get mean neighbors and their neighbors
            if j == np.shape(grid)[1]-2 or i == np.shape(grid)[0]-2: continue
            m2 = m
            m2 += np.sum([grid[i-2,j-2], grid[i-1,j-2] , grid[i,j-2] , grid[i+1,j-2], grid[i+2,j-2]])
            m2 += np.sum([grid[i-2,j-1], grid[i+2,j-1]])
            m2 += np.sum([grid[i-2,j], grid[i+2,j] ])
            m2 += np.sum([grid[i-2,j+1], grid[i+2,j+1]])
            m2 += np.sum([grid[i-2,j+2], grid[i-1,j+2] , grid[i,j+2] , grid[i+1,j+2], grid[i+2,j+2]])
            grid_means2[i,j] = m2 / (16+8)
    return grid_sums, grid_means, grid_means2

def filter_nodule_generated2(grid1, mini_mask_not_lungs, grid_cells_active_, mini_patched_borders):
    '''Find the pixels where a nodule is generated. Remove the lung borders using dilation,
    then find the largest connected component. Filter the area of the largest component, 
    then place the filtered image back (remove the filtered borders [too dark]). Finally add the lung borders.'''
    # remove the lung borders. dilate and find the largest component
#     grid_change = grid1 > .2
#     grid_change = grid_change * (~mini_mask_not_lungs)
#     grid_change = ndimage.morphology.binary_dilation(grid_change)
#     grid_labeled, nr = ndimage.label(grid_change)
#     sizes = ndimage.sum(grid_change, grid_labeled,range(1,nr+1)) 
#     elem_max = (np.where(sizes==sizes.max())[0] + 1)[0]
#     largest_comp = grid_labeled == elem_max
    # filter the area of the largest component,
    # then place the filtered image back (remove the filtered borders [too dark])
    y,x = np.where(grid_cells_active_==1)
    y_max = np.max(y); y_min = np.min(y)
    x_max = np.max(x); x_min = np.min(x)
    grid_filter = grid1[y_min:y_max,x_min:x_max]
    grid_filter = signal.wiener(grid_filter)
    grid_filter_added = copy(grid1)
    grid_filter_added[y_min:y_max,x_min:x_max] = grid_filter
    #borders
    grid_filter_added[y_min:y_max,x_min-1:x_min+1] = grid1[y_min:y_max,x_min-1:x_min+1]
    grid_filter_added[y_min:y_max,x_max-1:x_max+1] = grid1[y_min:y_max,x_max-1:x_max+1]
    grid_filter_added[y_min-1:y_min+1,x_min:x_max] = grid1[y_min-1:y_min+1,x_min:x_max]
    grid_filter_added[y_max-1:y_max+1,x_min:x_max] = grid1[y_max-1:y_max+1,x_min:x_max]
    # Add lung borders
    grid_filter_added = grid_filter_added*(~mini_mask_not_lungs)+mini_patched_borders*mini_mask_not_lungs
    return grid_filter_added

def pick_a_nodule_and_set_a_nodule_seed(path_data):
    ## pick a nodule
    # path_data = f'/data/OMM/project results/Jun 16 19 - Deep Image Prior2/v18/individual_nodules_for_automata/'
    files = os.listdir(f'{path_data}original/')
    files = np.sort(files)
    block_name = files[0]
    last, orig, mask, mask_lungs = get_nodule(path_data, block_name)
    auto_last, auto_orig, auto_mask, auto_mask_lungs, auto_patched, yy, xx = find_nodule_with_mask(last, orig, mask, mask_lungs)
    mini_orig, mini_mask, mini_mask_lungs, mini_patched, mask_lungs_min_y, mask_lungs_min_x, mini_last = coords_limits_using_mask_lungs(auto_last, auto_orig, auto_mask, auto_mask_lungs, auto_patched)
    ## add seed
    grid_cells_active = np.zeros_like(mini_patched)
    speckles_radious = 2
    speckles_amount = 8
    rand_y_coords = np.random.randint(-speckles_radious, speckles_radious, speckles_amount)
    rand_x_coords = np.random.randint(-speckles_radious, speckles_radious, speckles_amount)
    for rand_y, rand_x in zip(rand_y_coords, rand_x_coords):
        rand_int = random.uniform(0.1, 0.45)
        mini_patched[mask_lungs_min_y + yy + rand_y, mask_lungs_min_x + xx + rand_x] += rand_int
        grid_cells_active[mask_lungs_min_y + yy + rand_y, mask_lungs_min_x + xx + rand_x] += rand_int
    grid_cells_active = grid_cells_active > 0    
    mini_patched_clean = copy(mini_patched)
    ##
    mini_mask_not_lungs = -mini_mask_lungs+1
    mini_mask_not_lungs = ndimage.morphology.binary_dilation(mini_mask_not_lungs, iterations=3)
    mini_patched_borders = copy(mini_patched)
    mini_patched = np.maximum(mini_patched - (mini_mask_not_lungs*.3), np.zeros_like(mini_patched))
    return mini_patched, mini_orig, mini_last, mini_mask, mini_mask_not_lungs, mini_patched_clean, mini_patched_borders, grid_cells_active
    

def survive_and_birth_individual_list(neighbors, means, means2, values, survives, births, cells_active):
    neighbors = np.asarray(neighbors)
    means = np.asarray(means)
    values = np.asarray(values)
    survives = np.asarray(survives)
    births = np.asarray(births)
    
    grid_new = copy(values)
    for i in range(1,np.shape(neighbors)[0]-1):
        for j in range(1,np.shape(neighbors)[1]-1):
            
            # SURVIVE
            if .15 < grid_new[i,j] < .6 and cells_active[i,j] > 0: 
                grid_new[i,j] = np.squeeze([np.mean((means2[i,j], means[i,j], means[i,j], values[i,j] + .45)) if neighbors[i,j] in survives[i,j] else values[i,j]-.1])
                #grid_new[i,j] = np.squeeze([values[i,j]+.1 if neighbors[i,j] in survives[i,j] else values[i,j]-.1])
            # BIRTH
            elif grid_new[i,j] < .15 and cells_active[i,j] > 0:
                rand_close_to_alive = random.gauss(0.45,.05)
                last_cell_state = grid_new[i,j]
                if neighbors[i,j] in births[i,j]:
                    grid_new[i,j] = np.mean((rand_close_to_alive, means[i,j], last_cell_state))
            # LIMIT
            elif grid_new[i,j] > .6:
                rand_close_to_alive = random.gauss(0.55,.05)
                grid_new[i,j] = rand_close_to_alive
            else: pass
    
    return grid_new

def filter_nodule_generated3(grid1, mini_mask_not_lungs, grid_cells_active_):
    '''Use grid_cells_active to find the pixels where a nodule is generated. 
    Filter the area where a nodule is generated
    Downsample and then upsample to remove the pixelated effect
    Normalize the down/upsampeld area to its original intensity range 
    then place the filtered image back (remove the filtered borders [too dark]). 
    Finally add the lung borders using mini_mask_not_lungs.'''
    y,x = np.where(grid_cells_active_==1)
    y_max = np.max(y); y_min = np.min(y)
    x_max = np.max(x); x_min = np.min(x)
    grid_filter = grid1[y_min:y_max,x_min:x_max]
    grid_filter = signal.wiener(grid_filter)
    #Adding downsampling -> upsampling and normalize to the original intensity range
    shape1, shape2 =  np.shape(grid_filter)
    original_min = np.min(grid_filter)
    original_range = np.max(grid_filter) - original_min
    
    grid_filter_downsampled = block_reduce(grid_filter,(2,2))
    shape_down1, shape_down2 =  np.shape(grid_filter_downsampled)
    grid_filter = ndimage.zoom(grid_filter_downsampled,(shape1/shape_down1, shape2/shape_down2))
    value_min = np.min(grid_filter)
    value_max = np.max(grid_filter)
    grid_filter = (grid_filter - value_min) / (value_max - value_min)
    grid_filter = grid_filter * original_range + original_min
    
    #
    grid_filter_added = copy(grid1)
    grid_filter_added[y_min:y_max,x_min:x_max] = grid_filter
    #borders
    grid_filter_added[y_min:y_max,x_min-1:x_min+1] = grid1[y_min:y_max,x_min-1:x_min+1]
    grid_filter_added[y_min:y_max,x_max-1:x_max+1] = grid1[y_min:y_max,x_max-1:x_max+1]
    grid_filter_added[y_min-1:y_min+1,x_min:x_max] = grid1[y_min-1:y_min+1,x_min:x_max]
    grid_filter_added[y_max-1:y_max+1,x_min:x_max] = grid1[y_max-1:y_max+1,x_min:x_max]
    # Add lung borders
    grid_filter_added = grid_filter_added*(~mini_mask_not_lungs)+mini_patched_borders*mini_mask_not_lungs
    return grid_filter_added

def survive_and_birth_individual_list(neighbors, means, means2, values, survives, births, cells_active):
    neighbors = np.asarray(neighbors)
    means = np.asarray(means)
    values = np.asarray(values)
    survives = np.asarray(survives)
    births = np.asarray(births)
    
    grid_new = copy(values)
    for i in range(1,np.shape(neighbors)[0]-1):
        for j in range(1,np.shape(neighbors)[1]-1):
            
            # SURVIVE
            if .15 < grid_new[i,j] < .6 and cells_active[i,j] > 0: 
                grid_new[i,j] = np.squeeze([np.mean((means2[i,j], means[i,j], means[i,j], values[i,j] + .45)) if neighbors[i,j] in survives[i,j] else values[i,j]-.1])
                #grid_new[i,j] = np.squeeze([values[i,j]+.1 if neighbors[i,j] in survives[i,j] else values[i,j]-.1])
            # BIRTH
            elif grid_new[i,j] < .15 and cells_active[i,j] > 0:
                rand_close_to_alive = random.gauss(0.45,.05)
                last_cell_state = grid_new[i,j]
                if neighbors[i,j] in births[i,j]:
                    grid_new[i,j] = np.mean((rand_close_to_alive, means[i,j], last_cell_state))
            # LIMIT
            elif grid_new[i,j] > .6:
                rand_close_to_alive = random.gauss(0.55,.05)
                grid_new[i,j] = rand_close_to_alive
            else: pass
    
    return grid_new

def normalizePatches(npzarray):
    npzarray = npzarray
    
    maxHU = 400.
    minHU = -1000.
    
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

def denormalizePatches(npzarray):
    
    maxHU = 400.
    minHU = -1000.
 
    npzarray = (npzarray * (maxHU - minHU)) + minHU
    npzarray = (npzarray).astype('int16')
    return npzarray

def pick_a_raw_nodule_and_set_a_nodule_seed(path_data, block_name):
    ## pick a nodule
    # path_data = f'/data/OMM/project results/Jun 16 19 - Deep Image Prior2/v18/individual_nodules_for_automata/'
    last, orig, mask, mask_lungs = get_raw_nodule(path_data, block_name)
    auto_last, auto_orig, auto_mask, auto_mask_lungs, auto_patched, yy, xx = get_central_slice(last, orig, mask, mask_lungs)
    
    mini_patched = auto_patched
    mini_mask = auto_mask
    mini_mask_lungs = auto_mask_lungs
    mini_orig = auto_orig
    mini_last = auto_last
    mini_last2 = copy(mini_last)
    
    ## add seed
    grid_cells_active = np.zeros_like(mini_patched)
    speckles_radious = 2
    speckles_amount = 8
    rand_y_coords = np.random.randint(-speckles_radious, speckles_radious, speckles_amount)
    rand_x_coords = np.random.randint(-speckles_radious, speckles_radious, speckles_amount)
    for rand_y, rand_x in zip(rand_y_coords, rand_x_coords):
        rand_int = random.uniform(0.1, 0.45)
        mini_patched[rand_y + yy, rand_x + xx] += rand_int
        grid_cells_active[rand_y + yy, -rand_x + xx] += rand_int
    
    grid_cells_active = grid_cells_active > 0    
    mini_patched_clean = copy(mini_patched)
    ##
    mini_mask_not_lungs = -mini_mask_lungs+1
    mini_mask_not_lungs = ndimage.morphology.binary_dilation(mini_mask_not_lungs, iterations=3)
    mini_patched_borders = copy(mini_patched)
    mini_patched = np.maximum(mini_patched - (mini_mask_not_lungs*.3), np.zeros_like(mini_patched))
    return mini_patched, mini_orig, mini_last2, mini_mask, mini_mask_not_lungs, mini_patched_clean, mini_patched_borders, grid_cells_active

def get_raw_nodule(path_data, block_name):
    '''Read the original, inpainted inserted and masks. The previous implementation (on LIDC not LUNA)
    also read the mask lungs. Now it is not available. Therefore we make one'''
    path_last = f'{path_data}inpainted inserted/'
    path_orig = f'{path_data}original/'
    path_mask_lungs = f'{path_data}mask lungs/'
    path_mask = f'{path_data}mask/'
    
    last = np.fromfile(f'{path_last}{block_name}',dtype='int16').astype('float32').reshape((64,64,64))
    orig = np.fromfile(f'{path_orig}{block_name}',dtype='int16').astype('float32').reshape((64,64,64))
    mask = np.fromfile(f'{path_mask}{block_name}',dtype='int16').astype('float32').reshape((64,64,64))
    
    # Normalize
    last = normalizePatches(last)
    orig = normalizePatches(orig)
    
    # Make mask lungs
    mask_lungs = np.ones_like(orig)
    z,y,x = np.where(orig==np.min(orig))
    mask_lungs[z,y,x] = 0
    
    return last, orig, mask, mask_lungs

def survive_and_birth(neighbors, means, values, survive, birth, mini_mask_dilated_twice):
    grid_new = copy(values)
    y,x = np.where(mini_mask_dilated_twice==1)
    
    for i in range(1,np.shape(neighbors)[0]-1):
        if i in y:
            for j in range(1,np.shape(neighbors)[1]-1):
                if j in x:

                    # SURVIVE
                    if neighbors[i,j] in survive:
                        if .3 < grid_new[i,j] < .6: 
                            grid_new[i,j] = np.squeeze([values[i,j]+.1])
    
                    # BIRTH
                    if neighbors[i,j] in birth:
                        if grid_new[i,j] < .2:
                            rand_close_to_alive = random.gauss(0.55,.05)
                            last_cell_state = grid_new[i,j]
                            grid_new[i,j] = np.mean((rand_close_to_alive, means[i,j], last_cell_state))
                    
                    # LIMIT
                    elif grid_new[i,j] > .6:
                        rand_close_to_alive = random.gauss(0.55,.05)
                        grid_new[i,j] = rand_close_to_alive
                    else: pass
    ## format survive and birth texts
    survive_text = [str(i) for i in survive]
    survive_text = ''.join(survive_text)
    birth_text = [str(i) for i in birth]
    birth_text = ''.join(birth_text)
    
    return grid_new, survive_text, birth_text

def plot_inpain_orig_seed_generated(path_, mini_last, mini_orig, mini_patched_clean, grid_filtered):
    fig, ax = plt.subplots(2,2, figsize=(8,8.4))
    ax[0,0].imshow(mini_last)
    ax[0,0].set_title('inpain', fontsize=16)
    ax[0,1].imshow(mini_orig)
    ax[0,1].set_title('orig', fontsize=16)
    ax[1,0].imshow(mini_patched_clean)
    ax[1,0].set_title('seed', y=-0.07, fontsize=16)
    ax[1,1].imshow(grid_filtered)
    ax[1,1].set_title('generated', y=-0.07, fontsize=16)
    for axx in ax.ravel(): axx.axis('off')
    plt.tight_layout()
    if path_ != 'no':
        plt.savefig(path_)
        plt.close()

def plot_22(mini_last, mini_orig, mini_patched_clean, grid_filtered, central_slice=-1, plot_size=8, vmin=0, vmax=1):
    if central_slice != -1:
        mini_last = mini_last[central_slice]
        mini_orig = mini_orig[central_slice]
        mini_patched_clean = mini_patched_clean[central_slice]
        grid_filtered = grid_filtered[central_slice]
    fig, ax = plt.subplots(2,2, figsize=(plot_size,plot_size))
    ax[0,0].imshow(mini_last, vmin=vmin, vmax=vmax)
    ax[0,1].imshow(mini_orig, vmin=vmin, vmax=vmax)
    ax[1,0].imshow(mini_patched_clean, vmin=vmin, vmax=vmax)
    ax[1,1].imshow(grid_filtered, vmin=vmin, vmax=vmax)
    for axx in ax.ravel(): axx.axis('off')
    plt.tight_layout()

# 3D ====================

def count_neighbors_and_get_means_3D_mask(grid, mask, threshold = 0.5):
    '''Get the sum and and mean of the cells around a cell WITHOUT MULTIPROCESSING'''
    grid_sums = np.zeros_like(grid)
    grid_means = np.zeros_like(grid)
    grid_means2 = np.zeros_like(grid)
    
    z,y,x = np.where(mask==1)
    
    for i in range(1,np.shape(grid)[0]-1):
        if i in z:
            for j in range(1,np.shape(grid)[1]-1):
                if j in y:
                    for k in range(1,np.shape(grid)[2]-1):
                        if k in x:
                            #print(i,j)
                            # count alive neighbors
                            s = 0
                            s += np.sum([grid[i-1,j-1, k-1] > threshold, grid[i,j-1, k-1] > threshold, grid[i+1,j-1, k-1] > threshold])
                            s += np.sum([grid[i-1,j, k-1] > threshold, grid[i,j, k-1] > threshold, grid[i+1,j, k-1] > threshold])
                            s += np.sum([grid[i-1,j+1, k-1] > threshold, grid[i,j+1, k-1] > threshold, grid[i+1,j+1, k-1] > threshold])

                            s += np.sum([grid[i-1,j-1, k] > threshold, grid[i,j-1, k] > threshold, grid[i+1,j-1, k] > threshold])
                            s += np.sum([grid[i-1,j, k] > threshold, grid[i+1,j, k] > threshold])
                            s += np.sum([grid[i-1,j+1, k] > threshold, grid[i,j+1, k] > threshold, grid[i+1,j+1, k] > threshold])

                            s += np.sum([grid[i-1,j-1, k+1] > threshold, grid[i,j-1, k+1] > threshold, grid[i+1,j-1, k+1] > threshold])
                            s += np.sum([grid[i-1,j, k+1] > threshold, grid[i,j, k+1] > threshold, grid[i+1,j, k+1] > threshold])
                            s += np.sum([grid[i-1,j+1, k+1] > threshold, grid[i,j+1, k+1] > threshold, grid[i+1,j+1, k+1] > threshold])

                            grid_sums[i,j,k] = s
                            # get mean neighbors
                            m = 0
                            m += np.sum([grid[i-1,j-1, k-1], grid[i,j-1, k-1], grid[i+1,j-1, k-1]])
                            m += np.sum([grid[i-1,j, k-1], grid[i,j, k-1], grid[i+1,j, k-1]])
                            m += np.sum([grid[i-1,j+1, k-1], grid[i,j+1, k-1], grid[i+1,j+1, k-1]])

                            m += np.sum([grid[i-1,j-1, k], grid[i,j-1, k], grid[i+1,j-1, k]])
                            m += np.sum([grid[i-1,j, k], grid[i+1,j, k]])
                            m += np.sum([grid[i-1,j+1, k], grid[i,j+1, k], grid[i+1,j+1, k]])

                            m += np.sum([grid[i-1,j-1, k+1], grid[i,j-1, k+1], grid[i+1,j-1, k+1]])
                            m += np.sum([grid[i-1,j, k+1], grid[i,j, k+1], grid[i+1,j, k+1]])
                            m += np.sum([grid[i-1,j+1, k+1], grid[i,j+1, k+1], grid[i+1,j+1, k+1]])
                            grid_means[i,j,k] = m / 8
                            # get mean neighbors and their neighbors
                            # 
    return grid_sums, grid_means

def add_seed_3D(last, zz = 31, yy = 31, xx = 31, speckles_radious = 2, speckles_amount = 8):
    mini_patched = copy(last)
    grid_cells_active = np.zeros_like(mini_patched)
    rand_y_coords = np.random.randint(-speckles_radious, speckles_radious, speckles_amount)
    rand_x_coords = np.random.randint(-speckles_radious, speckles_radious, speckles_amount)
    rand_z_coords = np.random.randint(-speckles_radious, speckles_radious, speckles_amount)
    for rand_y, rand_x, rand_z in zip(rand_y_coords, rand_x_coords, rand_z_coords):
        rand_int = random.uniform(0.1, 0.45)
        mini_patched[zz + rand_z, yy + rand_y, xx + rand_x] += rand_int
        grid_cells_active[zz + rand_z, yy + rand_y, xx + rand_x] = 1
        
    return mini_patched, grid_cells_active
@jit
def survive_and_birth_individual_list_3D_mask(neighbors, means, mask, values, survives, births, cells_active, grid_new):
    neighbors = np.asarray(neighbors)
    means = np.asarray(means)
    values = np.asarray(values)
    survives = np.asarray(survives)
    births = np.asarray(births)
    LIMIT_UP = 0.75
    LIMIT_DOWN = 0.15
    grid_new = np.asarray(grid_new)
    
    z,y,x = np.where(mask==1)
    
    # grid_new = copy(values)
    for i in range(1,np.shape(neighbors)[0]-1):
        if i in z:
            for j in range(1,np.shape(neighbors)[1]-1):
                if j in y:
                    for k in range(1,np.shape(neighbors)[2]-1):
                        if k in x:

                            # SURVIVE
                            if LIMIT_DOWN < grid_new[i,j,k] < LIMIT_UP and cells_active[i,j,k] > 0: 
                                grid_new[i,j,k] = [(means[i,j,k] + values[i,j,k]+.05)/3 if neighbors[i,j,k] in survives else values[i,j,k]-.05][0]
                            # BIRTH
                            elif grid_new[i,j,k] < LIMIT_DOWN and cells_active[i,j,k] > 0:
                                rand_close_to_alive = 0.2
                                last_cell_state = grid_new[i,j,k]
                                if neighbors[i,j,k] in births:
                                    grid_new[i,j,k] = (rand_close_to_alive + means[i,j,k] + last_cell_state)/3
                            # LIMIT
                            elif grid_new[i,j,k] > LIMIT_UP:
                                rand_close_to_alive = (values[i,j,k]-.5 + means[i,j,k])/2
                                grid_new[i,j,k] = rand_close_to_alive
                            else: pass
    
    return grid_new

def filter_nodule_generated3_3D(grid1, grid_cells_active_):
    '''Use grid_cells_active to find the pixels where a nodule is generated. 
    Filter the area where a nodule is generated
    Downsample and then upsample to remove the pixelated effect
    Normalize the down/upsampeld area to its original intensity range 
    then place the filtered image back (remove the filtered borders [too dark]). 
    Finally add the lung borders using mini_mask_not_lungs.
    v3_1: 
    - instead of copying the filtered rectangle [y_min:y_max,x_min:x_max] we only copy the values[y,x]
    - corrected  the limits of the area that will be filtered, now we include x/y_max+1 to include the last pxl'''
    z,y,x = np.where(grid_cells_active_==1)
    y_max = np.max(y); y_min = np.min(y)
    x_max = np.max(x); x_min = np.min(x)
    z_max = np.max(z); z_min = np.min(z)
    grid_filter = grid1[z_min:z_max+1,y_min:y_max+1,x_min:x_max+1]
    grid_filter = signal.wiener(grid_filter)
    #Adding downsampling -> upsampling and normalize to the original intensity range
    shape1, shape2, shape3 =  np.shape(grid_filter)
    original_min = np.min(grid_filter)
    original_range = np.max(grid_filter) - original_min
    
    grid_filter_downsampled = block_reduce(grid_filter,(2,2,2))
    shape_down1, shape_down2, shape_down3 =  np.shape(grid_filter_downsampled)
    grid_filter = ndimage.zoom(grid_filter_downsampled,(shape1/shape_down1, shape2/shape_down2, shape3/shape_down3))
    value_min = np.min(grid_filter)
    value_max = np.max(grid_filter)
    grid_filter = (grid_filter - value_min) / (value_max - value_min)
    grid_filter = grid_filter * original_range + original_min
    
    #
    grid_filter_added = copy(grid1)
    #grid_filter_added[y_min:y_max,x_min:x_max] = grid_filter
    y2 = y-y_min
    x2 = x-x_min
    z2 = z-z_min
    grid_filter_added[z,y,x] = grid_filter[z2,y2,x2]
    return grid_filter_added

def figure_action(action_proba, n_actions, step, folder, save=True):
    '''make figure that shows that values of the action vector for the cellular 
    automata. One row for survive and one row for birth values'''
    fig, ax = plt.subplots(2,1,figsize=(20,2.5))
    larger_than_1_survive = 'k'
    larger_than_1_birth = 'k'
    if np.max(action_proba[:n_actions])>1: larger_than_1_survive = 'r'
    if np.max(action_proba[n_actions:])>1: larger_than_1_birth = 'r'
    g0 = sns.heatmap(np.expand_dims(action_proba[:n_actions],0), ax=ax[0], vmin=0, vmax=1, cmap='viridis', cbar=False, linewidths=.1, linecolor=larger_than_1_survive, yticklabels=['survive'])
    g1 = sns.heatmap(np.expand_dims(action_proba[n_actions:],0), ax=ax[1], vmin=0, vmax=1, cmap='viridis', cbar=False, linewidths=.1, linecolor=larger_than_1_birth, yticklabels=['birth'])
    g0.set_yticklabels(g0.get_yticklabels(), rotation = 0);
    g1.set_yticklabels(g1.get_yticklabels(), rotation = 0);
    ax[0].set_title(f'the outputs might not be in order_{step:05d}', fontsize=18);
    fig.tight_layout()
    if save:
        fig.savefig(f'{folder}action_{step:05d}.png')
        plt.close()

def figure_make_gif(folder, new_name):
    '''Make gif from images saved on a folder AND DELETE THE IMAGES.
    Made to work with figure_action'''
    images = []
    filenames = os.listdir(folder)
    filenames = np.sort(filenames)
    for idx, filename in enumerate(filenames):
        images.append(imageio.imread(f'{folder}{filename}'))
        if idx==len(filenames)-1:
            for i in range(10):
                images.append(imageio.imread(f'{folder}{filename}'))
        os.remove(f'{folder}{filename}')
    imageio.mimsave(new_name, images)

def figure_a_few_original_and_augmented_nodules(preds, skip, total_cols=5):
    '''For each (5) column plot a nodule and on each subsequent row plot the
    images generated for that nodule by the cellular automata'''
    total_cols=total_cols
    max_rows=1
    # Get the indices where a new nodule is selected
    pix_active = [np.sum(p>0) for p in preds]    
    pix_active_diff = np.diff(pix_active)
    ndl_new_idx = np.where(pix_active_diff < -500)[0]+1
    ndl_new_idx = np.insert(ndl_new_idx,0,0)
    idx_len_temps=[]
    # Get the maximum number of rows we should include
    for idx, i in enumerate(ndl_new_idx[skip:]):
        if idx == total_cols: break
        idx_this, idx_next = ndl_new_idx[skip:][idx], ndl_new_idx[skip:][idx+1]
        idx_len_temp = idx_next - idx_this
        if idx_len_temp > max_rows: 
            max_rows = idx_len_temp
        idx_len_temps.append(idx_len_temp)
    # For each column plot a nodule and its augmented versions
    fig, ax = plt.subplots(max_rows, total_cols,figsize=(12,12*(.2*max_rows)))
    # each nodule in a separate column
    for idx, i in enumerate(ndl_new_idx[skip:]):
        if idx == total_cols: break
        # Get start and end of each nodule
        idx_this, idx_next = ndl_new_idx[skip:][idx], ndl_new_idx[skip:][idx+1]
        # Increase steps in rows
        for iidx in range(max_rows):
            z,y,x = np.where(preds[idx_this+iidx][0]>0)
            zz = int(np.median(z))
            if iidx==0: 
                orig = preds[idx_this+iidx][0]
                text_corner=f'size={np.sum(preds[idx_this+iidx][0]>0):.0f}'
            else: text_corner=f'{np.sum(np.abs(preds[idx_this+iidx][0] - orig)):.0f}'
            if iidx < idx_len_temps[idx]:
                ax[iidx,idx].imshow(preds[idx_this+iidx][0][zz], vmin=0, vmax=1)
                ax[iidx,idx].text(1,3,text_corner,c='y', fontsize=14)
            else:
                ax[iidx,idx].imshow(np.zeros_like(preds[0][0][zz]))
    for axx in ax.ravel(): axx.axis('off')
    fig.tight_layout()

def figure_a_few_original_and_augmented_nodules_rows(preds, skip, total_cols=5):
    '''For each (5) column plot a nodule and on each subsequent row plot the
    images generated for that nodule by the cellular automata'''
    total_cols=total_cols
    max_rows=1
    # Get the indices where a new nodule is selected
    pix_active = [np.sum(p>0) for p in preds]    
    pix_active_diff = np.diff(pix_active)
    ndl_new_idx = np.where(pix_active_diff < -500)[0]+1
    ndl_new_idx = np.insert(ndl_new_idx,0,0)
    idx_len_temps=[]
    # Get the maximum number of rows we should include
    for idx, i in enumerate(ndl_new_idx[skip:]):
        if idx == total_cols: break
        idx_this, idx_next = ndl_new_idx[skip:][idx], ndl_new_idx[skip:][idx+1]
        idx_len_temp = idx_next - idx_this
        if idx_len_temp > max_rows: 
            max_rows = idx_len_temp
        idx_len_temps.append(idx_len_temp)
    # For each column plot a nodule and its augmented versions
    skip_text=skip
    fig, ax = plt.subplots(total_cols, max_rows,figsize=(12*(.2*max_rows),12))
    # each nodule in a separate column
    for idx, i in enumerate(ndl_new_idx[skip:]):
        skip_text+=1
        if idx == total_cols: break
        # Get start and end of each nodule
        idx_this, idx_next = ndl_new_idx[skip:][idx], ndl_new_idx[skip:][idx+1]
        # Increase steps in rows
        for iidx in range(max_rows):
            z,y,x = np.where(preds[idx_this+iidx][0]>0)
            zz = int(np.median(z))
            if iidx==0: 
                orig = preds[idx_this+iidx][0]
                text_corner=f'size={np.sum(preds[idx_this+iidx][0]>0):.0f}'
            else: text_corner=f'{np.sum(np.abs(preds[idx_this+iidx][0] - orig)):.0f}'
            if iidx < idx_len_temps[idx]:
                ax[idx, iidx].imshow(preds[idx_this+iidx][0][zz], vmin=0, vmax=1)
                ax[idx, iidx].text(1,3,text_corner,c='y', fontsize=14)
            else:
                ax[idx, iidx].imshow(np.zeros_like(preds[0][0][zz]))
                if iidx==max_rows-1:
                    ax[idx, iidx].text(10,15, f'it={skip_text}',c='y', fontsize=22)
    for axx in ax.ravel(): axx.axis('off')
    fig.tight_layout()