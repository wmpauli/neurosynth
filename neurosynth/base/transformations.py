""" Various transformations between coordinate frames, atlas spaces, etc. """

import numpy as np
from numpy import linalg
import logging

logger = logging.getLogger('neurosynth.transformations')

def transform(foci, mat):
    """ Convert coordinates from one space to another using provided
    transformation matrix. """
    t = linalg.pinv(mat)
    foci = np.hstack((foci, np.ones((foci.shape[0], 1))))
    return np.dot(foci, t)[:, 0:3]


def xyz_to_mat(foci, target='MNI', xyz_dims=None, mat_dims=None):
    """ Convert an N x 3 array of XYZ coordinates to matrix indices. """
    foci = np.hstack((foci, np.ones((foci.shape[0], 1))))
    origins_mat = {
        'MNI' : [45, 63, 36],
        'WHS' : [36.72, 96.64, 37.24]
    }
    if not xyz_dims:
        if target == "MNI": 
            xyz_dims = [-2, 2, 2]
        elif target == "WHS":
            xyz_dims = [.25, .25, .25]
    origin = origins_mat[target]
    mat = np.array([[1.0/xyz_dims[0], 0, 0, origin[0]], [0, 1.0/xyz_dims[1], 0, origin[1]], [0, 0, 1.0/xyz_dims[2], origin[2]]]).T
    result = np.dot(foci, mat)[:, ::-1]  # multiply and reverse column order
    return np.round_(result).astype(int)  # need to round indices to ints


def mat_to_xyz(foci, mat_dims=None, xyz_dims=None):
    """ Convert an N x 3 array of matrix indices to XYZ coordinates. """
    foci = np.hstack((foci, np.ones((foci.shape[0], 1))))
    mat = np.array([[-2, 0, 0, 90], [0, 2, 0, -126], [0, 0, 2, -72]]).T
    result = np.dot(foci, mat)[:, ::-1]  # multiply and reverse column order
    return np.round_(result).astype(int)  # need to round indices to ints


def t88_to_mni():
    """ Convert Talairach to MNI coordinates using the Lancaster transform.
    Adapted from BrainMap scripts; see http://brainmap.org/icbm2tal/
    Details are described in Lancaster et al. (2007)
    (http://brainmap.org/new/pubs/LancasterHBM07.pdf). """
    return np.array([[0.9254, 0.0024, -0.0118, -1.0207], [-0.0048, 0.9316, -0.0871, -1.7667], [0.0152, 0.0883,  0.8924, 4.0926], [0.0, 0.0, 0.0, 1.0]]).T

def bregma_to_whs():
    """ convert between bregma coordinates and whs coordinates using Wolfgang's
    transform (BTW, this is also documented in Coordinates_v1_vs_v1.01.pdf at http://software.incf.org/software/waxholm-space-atlas-of-the-sprague-dawley-rat-brain"""
    # return np.array([[-1.0, 0.0, 0.0, 0.08], [0.0, -1.0, 0.0, 1.17], [0.0, 0.0, -1.0, 7.5], [0.0, 0.0, 0.0, 1.0]]).T
    # return np.array([[-0.981128, 0.0403032, 0.0986429, -0.0167773], [0.0335147, 0.999714, 0.00883351, -0.584105],[ -0.108513, 0.00552509, -1.06724, 6.317],[0.0, 0.0, 0.0, 1.0]]).T
    return np.array([[-1.12053, 0.0403032, -0.00445326, 0.291627],
                     [0.036827, 0.999714, 0.0122884, -0.605691],
                     [0.00413442, 0.00552509, -1.07273, 7.32751],
                     [0, 0, 0, 1]]).T


#    return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0], [0.0, 0.0, 1.0, 0], [0.0, 0.0, 0.0, 1.0]]).T

def identity():
    """ Don't do anything """
    return np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]).T


class Transformer(object):

    """ The Transformer class supports transformations between different
    image spaces. """

    def __init__(self, transforms=None, target='MNI'):
        """ Initialize a transformer instance. """
        self.target = target
        self.transformations = transforms

    def add(self, name, mat):
        """ Add a named linear transformation. """
        self.transformations[name] = mat

    def apply(self, name, foci):
        """ Apply a named transformation to a set of foci.

        If the named transformation doesn't exist, return foci untransformed.
        """
        if name in self.transformations:
            return transform(foci, self.transformations[name])
        else:
            logger.info("No transformation named '%s' found; coordinates left untransformed." % name)
            return foci
