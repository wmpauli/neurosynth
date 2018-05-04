"""Some handy functionality to be used by the Neurosynth test suite"""

__author__ = 'Wolfgang M. Pauli'
__copyright__ = 'Copyright (c) 2014 Wolfgang Pauli'
__license__ = 'GNU GPL 3.0'

from os.path import dirname, join, pardir, sep as pathsep
from neurosynth.base.dataset import Dataset
from neurosynth.base.dataset import FeatureTable
from neurosynth.base import transformations
from neurosynth.base.imageutils import *
from neurosynth.base.mask import Masker
import os
import nibabel as nb
from neurosynth.analysis import meta

base_path = '/home/pauli/Development/neurobabel/'
test_data_path = base_path + 'ACE/'
masker_filename = base_path + 'atlases/whs_sd/WHS_SD_rat_brainmask_sm_v2.nii.gz'
atlas_filename = base_path + 'atlases/whs_sd/WHS_SD_rat_atlas_brain_sm_v2.nii.gz'
mask = nb.load(masker_filename)
masker = Masker(mask)
r = 1.0
transform = {'BREGMA': transformations.bregma_to_whs()}
target = 'WHS'

# load data set
# dataset = Dataset(os.path.join(test_data_path, 'db_bregma_export.txt'), masker=masker_filename, r=r, transform=transform, target=target)
# dataset.feature_table = FeatureTable(dataset)
# dataset.add_features(os.path.join(test_data_path, "db_bregma_features.txt")) # add features
# fn = dataset.get_feature_names()
dataset = Dataset(os.path.join(test_data_path, 'db_bregma_ns_vocab_export.txt'), masker=masker_filename, r=r, transform=transform, target=target)
dataset.feature_table = FeatureTable(dataset)
dataset.add_features(os.path.join(test_data_path, "db_bregma_ns_vocab_features.txt")) # add features
fn = dataset.get_feature_names()

from neurosynth.analysis import meta
feature = 'underpinnings'
ids = dataset.get_ids_by_features(('%s*' % feature), threshold=0.1)
ma = meta.MetaAnalysis(dataset, ids)
out_path = 'ma/%s' % feature
if not os.path.exists(out_path):
    os.mkdir(out_path)
    ma.save_results(out_path)


# for f in fn:
#     try:
#         ids = dataset.get_ids_by_features(f, threshold=0.001)
#     except TypeError:
#         print f
        
