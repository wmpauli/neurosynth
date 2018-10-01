from os import path

from neurosynth.base.dataset import Dataset
from neurosynth.base.dataset import FeatureTable
from neurosynth.base import transformations

def save_list(my_list, filename):
    with open(filename, 'w') as f:
        for s in my_list:
            f.write(str(s) + '\n')

def read_list(filename):
    with open(filename, 'r') as f:
        my_list = [int(line.rstrip('\n')) for line in f]
    return my_list
    

def get_interesting_features(dataset, ns_dataset, available_features, update=False, percentile=5):
    if os.path.exists('idx.txt') and update == False:
        idx = read_list('idx.txt')
    else:
        dsub = dataset.feature_table.data[available_features]
        m = dsub.mean(0)
        s = dsub.std(0)
        cv = s/m * 100
        b_idx = np.where(cv > np.percentile(cv, percentile))[0]
    
        dsub = ns_dataset.feature_table.data[available_features]
        m = dsub.mean(0)
        s = dsub.std(0)
        cv = s/m * 100
        ns_idx = np.where(cv > np.percentile(cv, percentile))[0]
    
        idx = list(set(b_idx).intersection(set(ns_idx)))

        save_list(idx, 'idx.txt')

    interesting_features = list(np.array(available_features)[idx])    
    interesting_features.sort()

    return interesting_features


def get_available_features(dataset, ns_dataset, threshold=0.001):
    fn = dataset.get_feature_names()
    available_features = []
    for f in fn:
        try:
            ns_ids = ns_dataset.get_ids_by_features(f, threshold=threshold)
            ids = dataset.get_ids_by_features(f, threshold=threshold)
            if len(ids) > 0 and len(ns_ids) > 0:
                available_features.append(f)
            else:
                print f
        except TypeError:
            print f

    available_features.sort()

    return available_features


def gen_feature_images(dataset, ns_dataset, available_features):
    """ create feature images for a dataset """ 
    for f in range(0, len(available_features)):
        feature = available_features[f]
        f_str = feature.replace(' ', '_')

        decoder = decode.Decoder(dataset=dataset, features=[feature])
        np.save('decoder/bregma/%s_feature_image.npy' % f_str, decoder.feature_images)

        ns_decoder = decode.Decoder(dataset=ns_dataset, features=[feature])
        np.save('decoder/neurosynth/%s_feature_image.npy' % f_str, ns_decoder.feature_images)
        print("%s, %f" % (feature, float(f)/len(available_features)))


resource_dir = path.join(path.pardir, 'resources') 

# make sure we have the data
dataset_dir = path.join(path.expanduser('~'), 'Documents', 'neurosynth-data')
database_path = path.join(dataset_dir, 'database_bregma.txt')
neurosynth_data_url = 'https://github.com/wmpauli/neurosynth-data'
if not path.exists(database_path):
    print("Please download dataset from %s and store it in %s" % (neurosynth_data_url, dataset_dir))


r = 1.0
transform = {'BREGMA': transformations.bregma_to_whs()}
target = 'WHS'
masker_filename = path.join(resource_dir, 'WHS_SD_rat_brainmask_sm_v2.nii.gz')

# load bregma dataset
dataset = Dataset(path.join(dataset_dir, 'database_bregma.txt'), masker=masker_filename, r=r, transform=transform, target=target)
dataset.feature_table = FeatureTable(dataset)
dataset.add_features(path.join(dataset_dir, "features_bregma.txt")) # add features

# load neurosynth dataset
ns_dataset = Dataset(path.join(dataset_dir, 'database_neurosynth.txt'))
ns_dataset.feature_table = FeatureTable(ns_dataset)
ns_dataset.add_features(path.join(dataset_dir, 'features_neurosynth.txt'))
ns_fn = ns_dataset.get_feature_names()

# create the intersection of vocabularies of the two fields (behavioral neuroscience and neuroimaging)
available_features = get_available_features(dataset, ns_dataset)

# let's remove unintereesting terms, to safe some storage space
interesting_features = get_interesting_features(dataset, ns_dataset, available_features, update=False, percentile=5)

# generate all the feature images for both datasets (this takes FOREVER, and A LOT of storage)
gen_feature_images(dataset, ns_dataset, available_features)

# initialize the transcoder, and save
rat_mask_file = '/home/pauli/Development/neurobabel/atlases/whs_sd/WHS_SD_rat_brainmask_sm_v2.nii.gz'
human_mask_file = '/usr/share/data/fsl-mni152-templates/MNI152_T1_2mm_brain.nii.gz'
masks = [rat_mask_file, human_mask_file]
transcoder = Transcoder(features=['decoder/bregma','decoder/neurosynth'], masks=masks, feature_names=interesting_features)
transcoder.save()

