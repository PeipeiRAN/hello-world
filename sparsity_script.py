import sys
import numpy as np
import matplotlib.pyplot as plt
from shared_functions import *
from sparsity_functions import *

import os
os.chdir('/Users/pran/Documents/code/new')

# INPUT DATA
data = np.load('example_image_stack.npy')
psf = np.load('example_psfs.npy')
psf_rot = rotate_stack(psf)

# SETUP
n_iter = 100
init_cost = 1e6
tolerance = 1e-5
n_iter_reweight = 3

# CONDAT
condat_sigma = condat_tau = 0.5
condat_relax = 0.5

# SPARSE
sparse_thresh = get_weight(data,psf) #[100,3,41]
#%%
print 'rho:', condat_relax
print 'sigma:', condat_sigma
print 'tau:', condat_tau
print ''
#print 'threshold:', sparse_thresh
#%%

# PROXIMIAL VARIABLES
x = np.copy(data)

data_shape = data.shape[-2:]
mr_filters = get_mr_filters(data_shape, opt=None, coarse= False)
dual_shape = [mr_filters.shape[0]]+ list(data.shape)
dual_shape[0], dual_shape[1] = dual_shape[1], dual_shape[0]
dual_shape = dual_shape
y = np.ones(dual_shape)
print dual_shape
#%%
# OPTIMISATION
costs = []
for iter_num_reweight in xrange(1, n_iter_reweight):
    
    print '-REWEIGHT:', iter_num_reweight + 1
    print ' '

    for iter_num in xrange(1, n_iter):

       grad = get_grad(x, data, psf, psf_rot)

       x_prox = prox_op(x - condat_tau * grad - condat_tau * linear_op_inv(y))

       y_temp = (y + condat_sigma *linear_op((2 * x_prox) - x))
#%%
       y_prox = (y_temp - condat_sigma *
              prox_dual_op_s(y_temp / condat_sigma, sparse_thresh / condat_sigma))

       x = condat_relax * x_prox + (1 - condat_relax) * x
       y = condat_relax * y_prox + (1 - condat_relax) * y
       tmp = get_cost(x, data, psf, sparse_thresh)
       costs.append(tmp)
       print iter_num, 'COST:', costs[-1]
       print ''

       if not iter_num % 4:
          cost_diff = np.linalg.norm(np.mean(costs[-4:-2]) - np.mean(costs[-2:]))
          print ' - COST DIFF:', cost_diff
          print ''

          if cost_diff < tolerance:
               print 'Converged!'
               break
       
    sparse_thresh = get_weight(x,psf)

# OUTPUT DATA

# DISPLAY SOME IMAGES
plt.figure(1)
plt.subplot(221)
plt.imshow(x[8], interpolation='nearest')
plt.subplot(222)
plt.imshow(x[18], interpolation='nearest')
plt.subplot(223)
plt.imshow(x[28], interpolation='nearest')
plt.subplot(224)
plt.imshow(x[38], interpolation='nearest')
plt.show()

# DISPLAY COST FUNCTION DECAY
plt.figure(2)
plt.plot(range(len(costs)), costs, 'r-')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()

