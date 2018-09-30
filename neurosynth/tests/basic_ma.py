# this script can be used to run basic forward and reverse inference on a feature
# the feature (e.g. fear) is expected to be supplied as a command line argument to the script

import sys

# assert that a feature has been supplied as command line argument. otherwise, exit
usage = 'Perform basic forward and reverse inference on a feature, e.g.: python %s <feature>' % sys.argv[0]
if len(sys.argv) < 2:
    print(usage)
    exit(1)

# the feature on which to perform forward and reverse inference
feature = sys.argv[1]
print("Performing forward and reverse inference for feature: %s" % feature)


# import modules
from os import path, makedirs
import nibabel as nb

from neurosynth.base.dataset import Dataset
from neurosynth.base.dataset import FeatureTable
from neurosynth.base import transformations
from neurosynth.base.imageutils import *
from neurosynth.base.mask import Masker
from neurosynth.analysis import meta


    
# path of WHS atlas files
resource_dir = path.join(path.pardir, 'resources')

# make sure we have the data
dataset_dir = path.join(path.expanduser('~'), 'Documents', 'neurosynth-data')
database_path = path.join(dataset_dir, 'database_bregma.txt')
neurosynth_data_url = 'https://github.com/wmpauli/neurosynth-data'
if not path.exists(database_path):
    print("Please download dataset from %s and store it in %s" % (neurosynth_data_url, dataset_dir))

# load dataset, both image table and feature table
r = 1.0 # 1mm smoothing kernel
transform = {'BREGMA': transformations.bregma_to_whs()}
target = 'WHS'
masker_filename = path.join(resource_dir, 'WHS_SD_rat_brainmask_sm_v2.nii.gz')
dataset = Dataset(path.join(dataset_dir, 'database_bregma.txt'), masker=masker_filename, r=r, transform=transform, target=target)
dataset.feature_table = FeatureTable(dataset)
dataset.add_features(path.join(dataset_dir, "features_bregma.txt")) # add features
fn = dataset.get_feature_names()

# get the ids of studies where this feature occurs
ids = dataset.get_ids_by_features(('%s*' % feature), threshold=0.1)
ma = meta.MetaAnalysis(dataset, ids)
results_path = path.join('results', 'meta', feature)
if not path.exists(results_path):
    makedirs(results_path)

    print("saving results to: %s" % results_path)
ma.save_results(results_path)

# note, figure 2 of manuscript was used by plotting the z-score statistical maps for forward inference (pAgF_z.nii.gz) and reverse inference (pFgA_z.nii.gz)
