# this script allows you to do some basic cross-species mapping.  For
# this script to work, you first need to download our version of the
# neurosynth data.  This is done in two steps (one small, and one big
# step):


# 1. git clone https://github.com/wmpauli/neurosynth-data.git
# 2. run the download_feature_images.sh script in the folder created above

from os import path 
from neurosynth.base.dataset import Dataset
from neurosynth.base.dataset import FeatureTable
from neurosynth.base import transformations
from neurosynth.base.imageutils import *
from neurosynth.base.mask import Masker
from neurosynth.analysis import meta, decode, transcode
import os, sys
import nibabel as nb
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn


def get_p_value(r, df):
    from scipy.special import betainc
    t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
    prob = betainc(0.5*df, 0.5, df / (df + t_squared))
    return prob

resource_dir = path.join(path.pardir, 'resources') 

# this might now be the most common use, but for demonstration
# purposes, we are going to have this script process arguments, so
# that we can quickly reproduce the figures of the manuscript
goal = sys.argv[1]
if goal == 'prelimbic':
    # we are using an anatomical masks, based on the the Paxinos atlas
    images_to_decode = [path.join(resource_dir, 'prelimbic.nii.gz')]
    direction = 'rat2human'
elif goal == 'frontolateral':
    # we are using an anatomical mask, based on the harvard-oxford anatomical atlas
    images_to_decode = [path.join(resource_dir, 'middle_frontal_gyrus.nii.gz')]
    direction = 'human2rat'
elif goal == 'fear':
    # we are using the results from a previously run reverse inference for the feature 'fear' in rodents
    images_to_decode = [path.join(resource_dir, 'fear_pFgA_z_FDR_0.01.nii.gz')]
    direction = 'rat2human'
elif goal == 'spatial_memory':
    # we are using the results from a previously run reverse inference for the feature 'fear' in rodents
    images_to_decode = [path.join(resource_dir, 'spatialMemory_pFgA_z_FDR_0.01.nii.gz')]
    direction = 'rat2human'
else:
    print('Please provide an argument for what you would like to do regarding cross-species mapping')
    exit(1)

# this is the main workhorse. This can be initialized different, for
# example by providing a list of folders and the names of features.
# Here, we are relying on a previously stored version, which is also
# faster. If you do want to start from scratch, start by running the
# script 'prepare_transcoder.py' in this directory
transcoder = transcode.Transcoder(source='from_arrays')

# Here, we are relying on a previously created feature_iamges, which
# is also faster. If you do want to start from scratch, start by
# running the script 'prepare_transcoder.py' in this directory.  This
# variable should point to the folder on your computer where you
# downloaded the feature images, ideally in your clone of the
# wmpauli/neurosynth-data repository
dataset_dir = path.join(os.path.expanduser('~'), 'Documents', 'neurosynth-data')


df = pd.DataFrame(columns=transcoder.feature_names)
top_features = []

for image_to_decode in images_to_decode:
    feature_vector, result = transcoder.transcode(image_to_decode, direction=direction)
    top_features += transcoder.get_top_features(feature_vector)
    df = df.append(pd.DataFrame(feature_vector.T, columns=df.columns))    
                    
df_s = df[top_features]

# create a plot for illustration purposes
ax = plt.subplot(111) 
for f in range(df_s.shape[0]):
    ax.plot(np.array(df_s.iloc[f,:]), label=None)
plt.xticks(np.arange(df_s.shape[1]), df_s.columns, rotation=45)
seaborn.despine()
plt.tight_layout()
plt.legend()
plt.show()

# perform FWE correction, by dividing by the number of voxels
result_fwe = result.copy()
df = len(feature_vector)
prob = get_p_value(result, df)
result_fwe[prob > .05 / result.shape[1]] = 0.0

# save results
result_path = os.path.join('results','transcoder')
if not path.exists(result_path):
    os.makedirs(result_path)

result_file = os.path.join(result_path, '%s.nii.gz' % goal)
print("Saving results to: %s" % result_file)
save_img(result_fwe, result_file, transcoder.maskers[transcoder.target_idx]) 

