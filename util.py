from scipy import misc
import numpy as np
import os
import re
import tensorflow as tf

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def read_names(filename):
    
    names = []
    with open(filename, 'r') as f:
        for line in f.readlines()[1:]:
            name = line.strip().split()
            names.append(name)
    return np.array(names)

def get_paths(lfw_dir, names, file_ext):
    nrof_skipped_names = 0

    path_list = []
    issame_list = []
    for name in names:
        if len(name) == 3:
            path0 = os.path.join(lfw_dir, name[0], name[0] + '_' + '%04d' % int(name[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, name[0], name[0] + '_' + '%04d' % int(name[2])+'.'+file_ext)
            issame = True
        if os.path.exists(path0) and os.path.exists(path1):
              # Only add the name if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
    
            nrof_skipped_names += 1

    if nrof_skipped_names>0:
        print('Skipped %d image names' % nrof_skipped_pairs)
    
    return path_list, issame_list


def load_data(image_paths, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        images[i,:,:,:] = img
    return images

def load_model(model_dir, meta_file, ckpt_file):
    model_dir_exp = os.path.expanduser(model_dir)
    saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
    saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))


def get_model_filenames(model_dir):
 
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file