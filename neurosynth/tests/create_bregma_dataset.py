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
masker_filename = base_path + 'atlases/whs_sd/WHS_SD_rat_one_sm_v2.nii.gz'
atlas_filename = base_path + 'atlases/whs_sd/WHS_SD_rat_atlas_brain_sm_v2.nii.gz'
mask = nb.load(masker_filename)
masker = Masker(mask)
r = 1.0
# transform = {'BREGMA': transformations.bregma_to_whs()}
#transform = {'BREGMA': transformations.identity()}
transform = {'BREGMA': transformations.bregma_to_whs()}
target = 'WHS'

# load data set
dataset = Dataset(os.path.join(test_data_path, 'db_bregma_export.txt'), masker=masker_filename, r=r, transform=transform, target=target)
dataset.feature_table = FeatureTable(dataset)
dataset.add_features(os.path.join(test_data_path, "db_bregma_features.txt")) # add features
fn = dataset.get_feature_names()

def get_whs_labels(filename=os.path.join(base_path, "atlases/whs_sd/WHS_SD_rat_atlas_v2.label")):
    ''' load the names of all labelled areas in the atlas (e.g. brainstem), return list of them '''
    in_file = open(filename, 'r')
    lines = in_file.readlines()
    labels = {}
    for line in lines:
        start = line.find("\"") + 1
        if start > 0:
            stop = line.find("\"", start)
            label = line[start:stop]
            idx = line.split()[0]
            labels[label] = int(idx)
    in_file.close()
    return labels

def get_featured_labels(labels, min_articles=10):
    ''' determine label ids of labels which occur as features in at least min_articles of studies '''
    featured_labels = dict()
    for label in labels.keys():
        try:
            ids = dataset.get_ids_by_features(('%s*' % label), threshold=0.001)
            n_ids = len(ids)
            if n_ids > min_articles:
                featured_labels[label] = [labels[label], n_ids]
        except:
            print("Problem with label: %s" % label)
    return featured_labels

def select_labels(labels, selection):
    ''' return label ids for selection of labels '''
    featured_labels = dict()
    for label in labels.keys():
        if label in selection:
            ids = dataset.get_ids_by_features(('%s*' % label), threshold=0.001)
            n_ids = len(ids)
            featured_labels[label] = [labels[label], n_ids]
    return featured_labels

def filter_atlas(atlas_filename, featured_labels):
    """
    return a version of an atlas nifti file, which only contains the labels of interest, but has all other
    brain regions ommitted.
    :param atlas_filename: filename of an atlas in nifti format
    :param featured_labels:
    :return:
    """
    atlas_nii = nb.load(atlas_filename)
    atlas = atlas_nii.get_data()
    filtered_atlas = np.zeros_like(atlas)
    for key in featured_labels.keys():
        label = featured_labels[key][0]
        idx = np.where(atlas == label)
        filtered_atlas[idx] = label
    return filtered_atlas


areas = get_whs_labels()
#featured_labels = get_featured_labels(areas, min_articles=30)
#featured_labels = select_labels(areas, ['fear']) #striatum', 'olfactory bulb', 'entorhinal cortex', 'brainstem'])
featured_labels = {'olfactory':[66], 'striatum':[30],'entorhinal':[114],'pallidus': [31],'thalamus': [39]}
filtered_atlas = filter_atlas(atlas_filename, featured_labels)
filtered_nii = nb.Nifti1Image(filtered_atlas, mask.get_affine())
filtered_nii.to_filename('filtered_atlas.nii.gz')

#save_img(filtered_atlas, 'filtered_atlas.nii.gz', dataset.masker)
# feature = "infralimbic"
#
# for feature in fn:
#     if feature.find(feature) > -1:
#         print feature
#
# ids = dataset.get_ids_by_features(('%s*' % feature), threshold=0.0001)
# print len(ids)

q = 0.01
res = np.zeros((len(featured_labels.keys()), dataset.image_table.data.shape[0]), dtype='double')
for index, feature in enumerate(featured_labels):
    print(feature)
    ids = dataset.get_ids_by_features(('%s*' % feature), threshold=0.001)
    ma = meta.MetaAnalysis(dataset, ids, q=q)
    res[index,:] = ma.images['pFgA_z_FDR_%s' % q]

winners = np.zeros(res.shape[1])
for index, feature in enumerate(featured_labels):
    # m = res.max(0)
    # non_zeros = np.where(m > 0)[0]
    # idx = np.where(res[index,:] == m)[0]
    # idx = list(set(non_zeros).intersection(idx))
    idx = np.where(res[index,:] > 0)[0]
    winners[idx] = featured_labels[feature][0]

segmentation_nii = nb.Nifti1Image(winners.reshape(80, 160, 80), mask.get_affine())
segmentation_nii.to_filename('neurobabel_seg.nii.gz')
# save_img(winners, 'neurobabel_seg.nii.gz', dataset.masker)

s = dataset.image_table.data.sum(1)
#s /= s.sum()
s = np.array(s)
segmentation_nii = nb.Nifti1Image(s.reshape(80, 160, 80), mask.get_affine())
save_img(s, 'all_20182014_1.nii.gz', dataset.masker)

# feature = 'striatum'
# winners = np.zeros(dataset.image_table.data.shape[0])
# ids = dataset.get_ids_by_features(('%s*' % feature), threshold=0.1)
# for i in ids:
#     a = dataset.get_image_data(i)
#     winners[np.where(a > 0)[0]] += 1
# winners /= winners.sum()
#
# save_img(winners, 'striatum.nii.gz', dataset.masker)
# save_img(a, '24672017.nii.gz', dataset.masker)



from neurosynth.analysis import meta
feature = 'prelimbic'
ids = dataset.get_ids_by_features(('%s*' % feature), threshold=0.1)
ma = meta.MetaAnalysis(dataset, ids)
out_path = 'ma/%s_12' % feature
if not os.path.exists(out_path):
    os.mkdir(out_path)
    ma.save_results(out_path)

    #
# freqs = np.log(dataset.image_table.data.sum(1) + 1)
#
# save_img(freqs, 'test.nii.gz', masker)


#dataset.save('test_dataset.npy')

#    dataset.add_features(test_data_path + 'test_features.txt')
#    return dataset
