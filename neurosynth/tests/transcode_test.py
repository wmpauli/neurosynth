# this script allows you to do some basic cross-species mapping.  For
# this script to work, you first need to download our version of the
# neurosynth data.  This is done in two step (one small, and one big
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
import os
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


# Here, we are relying on a previously created feature_iamges, which
# is also faster. If you do want to start from scratch, start by
# running the script 'prepare_transcoder.py' in this directory.  This
# variable should point to the folder on your computer where you
# downloaded the feature images, ideally in your clone of the
# wmpauli/neurosynth-data repository
dataset_dir = path.join(os.path.expanduser('~'), 'Documents', 'neurosynth-data')

# This can be initialized different, for example by providing a list
# of folders and the names of features.  
transcoder = transcode.Transcoder(source='from_arrays', dataset_dir=dataset_dir)

resource_dir = path.join(path.pardir, 'resources') 

# as an example, we will map the prelimbic cortex from rodents to humans
images_to_decode = [os.path.join(resource_dir, 'prelimbic.nii.gz')]
direction = 'rat2human'

# # alternatively, you could map the middle_frontal_gyrus to humans
# images_to_decode = [os.path.join(resource_dir, 'middle_frontal_gyrus.nii.gz')]
# direction = 'human2rat'

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
result_file = 'decoder/results/transcode_prelimbic.nii.gz'
save_img(result_1, result_file, transcoder.maskers[transcoder.target_idx]) 

