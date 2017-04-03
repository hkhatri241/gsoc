from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
#import numpy as np
import argparse
import facenet
import lfw
import util
import os
import sys
import math
import numpy as np
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
#from tensorflow.python.client import timeline
from panns import *


def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            # Read the file containing the pairs used for testing
            #readStart = time.clock()
            names = util.read_names(os.path.expanduser(args.lfw_names))
            #print(names)
            # Get the paths for the corresponding images
            paths, actual_issame = util.get_paths(os.path.expanduser(args.lfw_dir), names, args.lfw_file_ext)
            #readEnd = time.clock()
            print(paths)
           
            
            # Load the model
            #loadStart = time.clock()
            print('Model directory: %s' % args.model_dir)

            meta_file, ckpt_file = util.get_model_filenames(os.path.expanduser(args.model_dir))
            #run_metadata = tf.RunMetadata()
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            util.load_model(args.model_dir, meta_file, ckpt_file)
            #loadEnd = time.clock()
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]
            
            print('Image Size: %s' %str(image_size))
            print('Embedding Size: %s' %str(embedding_size))

            # Run forward pass to calculate embeddings
            print('Calculating embeddings')
            batch_size = args.lfw_batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
       
            runStart = time.clock()
            #run_metadata = tf.RunMetadata()
            for i in range(nrof_batches):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = util.load_data(paths_batch, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                break
            runEnd = time.clock()
           
            #print('Size of image list of batch 100: %d'%sizeImages)
            #print('Path array size : %d'%sys.getsizeof(paths))
            #print('Time to extract path from file: %d'%(readEnd - readStart))
            #print('Time to load model from disk: %d'%(loadEnd - loadStart))
            print('Time to calculate embeddings: %d'%(runEnd - runStart))

            buildIndexStart = time.clock()
            # create an index of Euclidean distance
            p = PannsIndex(dimension=128, metric='euclidean')
            for i in range(0,50):
               p.add_vector(emb_array[i][:])
            p.parallelize(True)
            p.build(40)
            
            buildIndexEnd = time.clock()
            
            results = p.query(emb_array[8][:], 4) #pick one face and find its 4 nearest neigbour
            print([paths[x[0]] for x in results] ) #putting brackets around generator expression makes it a list




            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.', default='~/facenet/lfw_align')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model_dir', type=str, 
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('--lfw_names', type=str,
        help='The file containing the names to use.', default='../data/pairs_copy.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))




