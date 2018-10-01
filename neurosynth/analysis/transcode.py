# emacs: -*- mode: python-mode; py-indent-offset: 2; tab-width: 2; indent-tabs-mode: nil -*-
# ex: set sts=2 ts=2 sw=2 et:
""" Decoding-related methods """

import numpy as np
from neurosynth.base.mask import Masker
from neurosynth.base import imageutils
from neurosynth.analysis import classify
from neurosynth.analysis import plotutils #import radar_factory
from neurosynth.base.imageutils import *
from os import path
import glob

class Transcoder:

    def __init__(self, dataset=None, method='pearson', features=None, masks=None, image_type='pFgA_z', threshold=0.001, source='from_folders', feature_names=None, animals = ['rat','human']):
        """ Initialize a new Decoder instance.

        Args:
          dataset: An optional Dataset instance containing features to use in decoding. (note: this is currently not sued at all)
          method: The decoding method to use (optional). By default, Pearson correlation.
          features: list of two filenames, one for each numpy file of feature_images, previously created with Decoder
          mask: An optional mask to apply to features and input images. If None, will use
            the one in the current Dataset.
          image_type: An optional string indicating the type of image to use when constructing
            feature-based images. See meta.analyze_features() for details. By default, uses 
            reverse inference z-score images.
          threshold: If decoding from a Dataset instance, this is the feature threshold to 
            use to generate the feature maps used in the decoding.


        """

        self.dataset = dataset
        self.set_masks(masks)
        self.animals = animals

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.load_feature_names()

        self.method = method.lower()

        self.load_features(features, feature_names, source=source, threshold=threshold)


    def set_masks(self, masks):
        resource_dir = path.join(path.pardir, 'resources') 
        if masks == None:
            masks = [path.join(resource_dir, 'WHS_SD_rat_brainmask_sm_v2.nii.gz'), path.join(resource_dir, 'MNI152_T1_2mm_brain.nii.gz')]
        self.maskers = []
        for m in range(0, len(masks)):
            self.maskers.append(Masker(masks[m]))


    def decode(self, images, save=None, round=4, names=None):
        """ Decodes a set of images.

        Args:
          images: The images to decode. Can be:
            - A single String specifying the filename of the image to decode
            - A list of filenames
            - A single NumPy array containing the image data
          save: Optional filename to save results to. If None (default), returns
            all results as an array.
          round: Optional integer indicating number of decimals to round result
            to. Defaults to 4.
          names: Optional list of names corresponding to the images in filenames.
            If passed, must be of same length and in same order as filenames.
            By default, the columns in the output will be named using the image
            filenames.

        Returns:
          An n_features x n_files numpy array, where each feature is a row and
          each image is a column. The meaning of the values depends on the
          decoding method used. """

        if isinstance(images, basestring) or isinstance(images, list):
            imgs_to_decode = imageutils.load_imgs(images, self.masker)
        else:
            imgs_to_decode = images

        methods = {
            'pearson': self._pearson_correlation(imgs_to_decode),
            # 'nb': self._naive_bayes(imgs_to_decode),
            'pattern': self._pattern_expression(imgs_to_decode)
        }

        result = np.around(methods[self.method], round)

        if save is not None:

            if names is None:
                if type(images).__module__ == np.__name__:
                    names = ['image_%d' for i in range(images.shape[1])]
                else:
                    names = images

            rownames = np.array(
                self.feature_names, dtype='|S32')[:, np.newaxis]

            f = open(save, 'w')
            f.write('\t'.join(['Feature'] + names) + '\n')
            np.savetxt(f, np.hstack((
                rownames, result)), fmt='%s', delimiter='\t')
        else:
            return result

    def set_method(self, method):
        """ Set decoding method. """
        self.method = method

    def load_features(self, features, feature_names, image_type=None, source='from_folders', threshold=0.001):
        """ Load features from current Dataset instance or a list of files. 
        Args:
            features: List containing paths to, or names of, features to extract. 
                Each element in the list must be a string containing either a path to an
                image, or the name of a feature (as named in the current Dataset).
                Mixing of paths and feature names within the list is not allowed.
            image_type: Optional suffix indicating which kind of image to use for analysis.
                Only used if features are taken from the Dataset; if features is a list 
                of filenames, image_type is ignored.
            from_folders: If True, the features argument is interpreted as a string pointing 
                to the location of a 2D ndarray on disk containing feature data, where
                rows are voxels and columns are individual features.
            threshold: If features are taken from the dataset, this is the threshold 
                passed to the meta-analysis module to generate fresh images.

        """
        if source == 'from_folders':
            self._load_features_from_folders(features, feature_names)
        # elif path.exists(features[0]):
        #     self._load_features_from_images(features)
        # else:
        #     self._load_features_from_dataset(features, image_type=image_type, threshold=threshold)
        elif source == 'from_arrays':
            self._load_features_from_arrays()
        else:
            print("Please specify 'source' to load feature images from!")


    def _load_features_from_arrays(self):
        self.feature_images = []
        for i, animal in enumerate(self.animals):
            filename = "feature_images_%s.npy" % animal
            try:
                feature_images = np.load(filename)
            except:
                print("Could not find file: %s" % filename)
            self.feature_images.append(feature_images)


    def _load_features_from_folders(self, features, feature_names, debug=False):
        """Load feature data from a 2D ndarrays on disk.  The assumption is
        that there are as many folders specified as there are animals
        specified.  Really, it should be two of each, because nothing
        else has been implemented.

        """

        if len(features) != len(self.animals):
            print("there is a mismatch between the number of animals and the number of image folders")
            return

        self.feature_images = [] # by the end of loading, this should have length of two (n animals)
        for d in range(0,len(features)):
            files = glob.glob(path.join(features[d], '*'))
            files.sort() # make sure to sort, because feature_names is sorted alphabetically
            tmp_file = np.load(files[0])
            feature_images = np.zeros((tmp_file.shape[0], len(feature_names)))
            f_valid = 0
            for f in range(0, len(files)):
                start = files[f].rfind('/') + 1
                end = files[f].find('_feature_image.npy', start)
                fn = files[f][start:end]
                fn = fn.replace('_', ' ')
                if fn in feature_names:
                    feature_images[:, f_valid] = np.load(files[f]).flatten()
                    f_valid += 1
                else:
                    if debug:
                        print("not including: %s, f: %s, f_v: %s, fn_v: %s" % (fn, f, f_valid, feature_names[f_valid]))
            self.feature_images.append(feature_images)

    # def _load_features_from_dataset(self, features=None, image_type=None, threshold=0.001):
    #     """ Load feature image data from the current Dataset instance. See load_features()
    #     for documentation.
    #     """
    #     self.feature_names = self.dataset.feature_table.feature_names
    #     if features is not None:
    #         self.feature_names = filter(lambda x: x in self.feature_names, features)
    #     from neurosynth.analysis import meta
    #     self.feature_images = meta.analyze_features(
    #         self.dataset, self.feature_names, image_type=image_type, threshold=threshold)
    #     # Apply a mask if one was originally passed
    #     if self.masker.layers:
    #         in_mask = self.masker.get_current_mask(in_global_mask=True)
    #         self.feature_images = self.feature_images[in_mask,:]

    # def _load_features_from_images(self, images, names=None):
    #     """ Load feature image data from image files.

    #     Args:
    #       images: A list of image filenames.
    #       names: An optional list of strings to use as the feature names. Must be
    #         in the same order as the images.
    #     """
    #     if names is not None and len(names) != len(images):
    #         raise Exception( "Lists of feature names and image files must be of same length!")
    #     self.feature_names = names if names is not None else images
    #     self.feature_images = imageutils.load_imgs(images, self.masker)

    def train_classifiers(self, features=None):
        ''' Train a set of classifiers '''
        # for f in features:
        #     clf = Classifier(None)
        #     self.classifiers.append(clf)
        pass
        
    
    # def _pattern_expression(self, imgs_to_decode):
    #     """ Decode images using pattern expression. For explanation, see:
    #     http://wagerlab.colorado.edu/wiki/doku.php/help/fmri_help/pattern_expression_and_connectivity
    #     """
    #     return np.dot(imgs_to_decode.T, self.feature_images).T


    def save(self):
        """ Save feature images, and available features """

        # save feature images
        self.save_feature_images()
        
        # save feature names
        self.save_feature_names()


    def save_feature_images(self):
        for i, feature_image in enumerate(self.feature_images):
            filename = "feature_images_%s.npy" % self.animals[i]
            np.save(filename, feature_image)


    def save_feature_names(self, filename = 'feature_names.txt'):
        """ save names of features that the transcoder is using """

        f = open(filename, 'w')
        f.write('\n'.join(self.feature_names) + '\n')
        f.close()


    def load_feature_names(self, filename = 'feature_names.txt'):
        f = open(filename,'r')
        self.feature_names = f.read().splitlines()
        f.close()

    def transcode(self, img_to_decode, direction='rat2human'):
        """ transcode image using Pearson's r.
    
        Computes the correlation between each input image and each feature image across
        voxels.
    
        Args:
        img_to_decode: An voxel x 1 numpy.ndarray of an image to transcode
        in columns.
        
        Returns:
        An n_features x n_images 2D array, with each cell representing the pearson
        correlation between the i'th feature and the j'th image across all voxels.
        """

        if direction == 'rat2human':
            self.source_idx, self.target_idx = [0,1]
            print("Projecting from Rat to Human Brain")
        else:
            self.source_idx, self.target_idx = [1,0]
            print("Projecting from Human to Rat Brain")

        if isinstance(img_to_decode, str):
            img_to_decode = load_imgs(img_to_decode, self.maskers[self.source_idx])

        x, y, z = img_to_decode.astype(float), self.feature_images[self.source_idx].astype(float), self.feature_images[self.target_idx].astype(float).T

        return self._pearson_correlation_sequence(x,y,z)


    def _pearson_correlation_sequence(self, x, y, z):
        """ calculate the sequence of correlations """

        x, y, z  = x - x.mean(0), y - y.mean(0), z - z.mean(0)
        x, y, z = x / np.sqrt((x ** 2).sum(0)), y / np.sqrt((y ** 2).sum(0)), z / np.sqrt((z ** 2).sum(0))
        
        feature_vector = x.T.dot(y).T
        feature_vector -= feature_vector.mean(0)
        feature_vector /= np.sqrt((feature_vector ** 2).sum(0))
        
        result = feature_vector.T.dot(z).T
        
        return feature_vector, result

    
    def get_top_features(self, feature_vector, n = 10):
        order = np.argsort(feature_vector.flatten())
        feature_names_ordered = np.array(self.feature_names)[order]

        start = len(self.feature_names) - n
        top_features = list(feature_names_ordered[start:])
        
        return top_features
