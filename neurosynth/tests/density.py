# this script can be used to create statistical maps showing for each voxel how often surgery was performed there 

from os import path
import nibabel as nib

from neurosynth.base.dataset import Dataset
from neurosynth.base import transformations
from neurosynth.base.imageutils import *
from neurosynth.base.mask import Masker


# directory where to store the results (output) of this script
results_dir = 'results'
if not path.exists(results_dir):
    os.makedirs(results_dir)

# path of WHS atlas files
resource_dir = path.join(path.pardir, 'resources')

# make sure we have the data
dataset_dir = path.join(os.path.expanduser('~'), 'Documents', 'neurosynth-data')
database_path = path.join(dataset_dir, 'database_bregma.txt')
neurosynth_data_url = 'https://github.com/wmpauli/neurosynth-data'
if not path.exists(database_path):
    print("Please download dataset from %s and store it in %s" % (neurosynth_data_url, dataset_dir))


# load data set
r = 1.0 # 1mm smoothing kernel
transform = {'BREGMA': transformations.bregma_to_whs()}
target = 'WHS'
masker_filename = path.join(resource_dir, 'WHS_SD_rat_brainmask_sm_v2.nii.gz')
dataset = Dataset(path.join(dataset_dir, 'database_bregma.txt'), masker=masker_filename, r=r, transform=transform, target=target)

# count for each voxel the number of studies reporting surgery there
s = dataset.image_table.data.sum(1)
s = np.array(s)

# save the counts to a nifti file
result_file = path.join(results_dir, 'count.nii.gz')
save_img(s, result_files, dataset.masker)

# convert frequencies to densities (i.e. normalize by dividing by the total number of studies
s /= s.sum()
result_file = path.join(results_dir, 'density.nii.gz')
save_img(s, result_file, dataset.masker)
