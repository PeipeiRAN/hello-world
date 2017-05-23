"""SHARED FUNCTIONS

Functions that do not depend on regularisation.

"""

import numpy as np
from scipy.signal import fftconvolve

def rotate(data):
    
    return np.rot90(data, 2)

def rotate_stack(data):
    
    return np.array([rotate(x) for x in data ])



def sigma_mad(a):
    '''Sigma Median Absolute deviation

    This method estimates the noise in the input a.
    data: Input data stack.

    '''

    return 1.4826 * np.median(np.abs(a - np.median(a)))


def prox_op(a):
    '''Proximity Operator

    This method returns only the positive elements from the input a.

    '''

    return a * (a > 0)


def hard(a, level):

    return a * (np.abs(a) >= level)


def soft(a, level):

    return np.sign(a) * (np.abs(a) - level) * (np.abs(a) >= level)


def convolve(a, b):
    '''Convolve

    Convolve input a with input b using FFT.

    '''

    return fftconvolve(a, b, mode='same')


def convolve_stack(a, b):
    '''Convolve Stack

    Convolve i with j for every i in input a and every j in input b.

    '''

    return np.array([convolve(i, j) for i, j in zip(a, b)])

def filter_convolve(data, filters, filter_rot=False):
    """Filter convolve

    This method convolves the input image with the wavelet filters

    Parameters
    ----------
    data : np.ndarray
        Input data, 2D array
    filters : np.ndarray
        Wavelet filters, 3D array
    filter_rot : bool, optional
        Option to rotate wavelet filters (default is 'False')

    Returns
    -------
    np.ndarray convolved data

    """

    if filter_rot:
        return np.sum((convolve(coef, f) for coef, f in
                      zip(data, rotate_stack(filters))), axis=0)

    else:
        return np.array([convolve(data, f) for f in filters])


def filter_convolve_stack(data, filters, filter_rot=False):
    """Filter convolve

    This method convolves the a stack of input images with the wavelet filters

    Parameters
    ----------
    data : np.ndarray
        Input data, 3D array
    filters : np.ndarray
        Wavelet filters, 3D array
    filter_rot : bool, optional
        Option to rotate wavelet filters (default is 'False')

    Returns
    -------
    np.ndarray convolved data

    """

    # Return the convolved data cube.
    return np.array([filter_convolve(x, filters, filter_rot=filter_rot)
                     for x in data])



def get_grad(x, y, psf, psf_rot):
    '''Get Gradient

    Calculate the gradient of the current deconvolution x, where
    grad(x) = M^T(Mx - y), M corresponds to colving each image with its PSF and
    M^T corresponds to convoving each image with its PSF rotated by 90 degrees.

    '''

    return convolve_stack(convolve_stack(x, psf) - y, psf_rot)


def cube2matrix(data_cube):
    """Cube to Matrix

    This method transforms a 3D cube to a 2D matrix

    Parameters
    ----------
    data_cube : np.ndarray
        Input data cube, 3D array

    Returns
    -------
    np.ndarray 2D matrix

    """

    return data_cube.reshape([data_cube.shape[0]] +
                             [np.prod(data_cube.shape[1:])]).T


def matrix2cube(data_matrix, im_shape):
    """Matrix to Cube

    This method transforms a 2D matrix to a 3D cube

    Parameters
    ----------
    data_matrix : np.ndarray
        Input data cube, 2D array
    im_shape : tuple
        2D shape of the individual images

    Returns
    -------
    np.ndarray 3D cube

    """

    return data_matrix.T.reshape([data_matrix.shape[1]] + list(im_shape))
