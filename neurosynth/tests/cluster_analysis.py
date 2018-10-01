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
import nibabel as nib
from neurosynth.analysis import meta
from time import time

# plotting functions
import matplotlib.pyplot as plt
import seaborn

# modules for cluster analysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn import metrics
from scipy import stats

base_path = '/home/pauli/Development/neurobabel/'
test_data_path = base_path + 'ACE/'
masker_filename = base_path + 'atlases/whs_sd/WHS_SD_rat_one_sm_v2.nii.gz'
atlas_filename = base_path + 'atlases/whs_sd/WHS_SD_rat_atlas_brain_sm_v2.nii.gz'
mask = nib.load(masker_filename)
masker = Masker(mask)
r = 1.0
transform = {'BREGMA': transformations.bregma_to_whs()}
target = 'WHS'

# load data set
dataset = Dataset(os.path.join(test_data_path, 'db_bregma_cog_atlas_export.txt'), masker=masker_filename, r=r, transform=transform, target=target)
dataset.feature_table = FeatureTable(dataset)
dataset.add_features(os.path.join(test_data_path, "db_bregma_cog_atlas_features.txt")) # add features
fn = dataset.get_feature_names()
features = dataset.get_feature_data()

n_xyz, n_articles = dataset.image_table.data.shape
# do topic modeling (LSA)
n_components = 20
svd = TruncatedSVD(n_components=n_components)
X = svd.fit_transform(features)
X_orig = X.copy()

X = StandardScaler().fit_transform(X_orig)

# db = DBSCAN(eps=10.0, min_samples=10).fit(X)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_

# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# print('Estimated number of clusters: %d' % n_clusters_)
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels))


true_k = 25
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=500, n_init=100, verbose=False, n_jobs=7, tol=1e-8)
# km = MiniBatchKMeans(n_clusters=true_k, n_init=100, batch_size=10)

print("Clustering sparse data with %s" % km)
t0 = time()
#km.fit(X)
km.fit(features)
print("done in %0.3fs" % (time() - t0))
print()

# this tells us for each study what cluster it loads on
labels = km.predict(features)

# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
# print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
# print("Adjusted Rand-Index: %.3f"
#        % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
       % metrics.silhouette_score(features, km.labels_, sample_size=1000))
# print("Silhouette Coefficient: %0.3f"
#        % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print()

# print("Top terms per cluster:")
# original_space_centroids = svd.inverse_transform(km.cluster_centers_)
# order_centroids = original_space_centroids.argsort()[:, ::-1]
# terms = features.columns
# for i in range(true_k):
#     print "Cluster %d:" % (i),
#     for ind in order_centroids[i, :10]:
#         print ' %s' % terms[ind],
#     print()

print("Top terms per cluster:")
#original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = features.columns
for i in range(true_k):
    print "Cluster %d:" % (i),
    for ind in order_centroids[i, :10]:
        print ' %s' % terms[ind],
    print()

    
# # needs to figure out how to find non-zero elements in sparse table    
# image_data= dataset.image_table.data.todense()
# for study in range(0,n_articles):
#     study_data = image_data[:,study]
#     idx = np.where(study_data > 0)[0]
#     label = labels[study] + 1
#     image_data[idx, study] = label

# # figure out for each voxel what the most common cluster assignment is
# modes = []
# for voxel in range(0,n_xyz):
#     vox_data = image_data[voxel,:].copy()
#     idx = np.where(vox_data > 0)[1]
#     if len(idx) > 0:
#         mode = int(stats.mode(vox_data[0,idx], axis=None)[0])
#     else:
#         mode = 0
#     modes.append(mode)
    
# modes = np.array(modes)

# out_nii = nib.Nifti1Image(np.reshape(modes, mask.shape), mask.affine)
# out_nii.to_filename('/tmp/tmp.nii.gz')

# needs to figure out how to find non-zero elements in sparse table    
image_data=dataset.image_table.data
cluster_results = np.zeros((true_k, n_xyz))
for study in range(0, n_articles):
    k = labels[study]
    study_data = image_data[:, study].todense()
    tf = np.array(study_data > 0).astype('int')
    cluster_results[k,:] += tf.flatten()

    sum = cluster_results.sum(1)
    sum = np.tile(sum, (n_xyz,1)).T
    cluster_results /= sum

out_nii = nib.Nifti1Image(np.reshape(cluster_results[4,:], mask.shape), mask.affine)
out_nii.to_filename('/tmp/tmp.nii.gz')
    
