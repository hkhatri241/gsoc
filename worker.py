import os, sys
import time

from celery.signals import worker_ready
from celery import Celery
import tensorflow as tf
import util
import math
import numpy as np
from panns import *

#model location and other variables 
lfw_dir = './data/lfw_align'
lfw_names = './data/pairs_copy.txt'
lfw_batch_size = 50
lfw_file_ext = 'png'
model_dir = './models'

#celery limits handlers on worker_init signal to 4 seconds. This is a turaround. Seems hackish. 
from celery.concurrency import asynpool
asynpool.PROC_ALIVE_TIMEOUT = 80.0 #set this long enough


app= Celery(backend='amqp', broker='amqp://')

#REMOVE THIS STUPIDITY LATER ON
emb_array = np.zeros((1170, 128))

@worker_ready.connect #This is to load the tensorflow model as soon as worker boots up.
def process_init( sender = None, conf=None, **kwargs):
    if sender.hostname == 'worker1@harshitpc':
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # Read the file containing the pairs used for testing
                #readStart = time.clock()
                names = util.read_names(os.path.expanduser(lfw_names))
                #print(names)
                # Get the paths for the corresponding images
                paths, actual_issame = util.get_paths(os.path.expanduser(lfw_dir), names, lfw_file_ext)
                #readEnd = time.clock()
                
                print("Done Initializing")
                # Load the model
                #loadStart = time.clock()
                print('Model directory: %s' % model_dir)

                meta_file, ckpt_file = util.get_model_filenames(os.path.expanduser(model_dir))
                #run_metadata = tf.RunMetadata()
                print('Metagraph file: %s' % meta_file)
                print('Checkpoint file: %s' % ckpt_file)
                util.load_model(model_dir, meta_file, ckpt_file)
                #loadEnd = time.clock()
                
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                
                image_size = 160 # Warning. This was hardcoded. General should be ---> images_placeholder.get_shape()[1]
                embedding_size = 128#Warning. This was hardcoded. General should be ---> embeddings.get_shape()[1]
                
                #print('Embedding Size: %s' %str(embedding_size))

                # Run forward pass to calculate embeddings
                print('Calculating embeddings')
                batch_size = lfw_batch_size
                nrof_images = len(paths)
                nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))

                #INCREDIBLY STUPID STUFF TO FOLLOW. WILL STRUCTURE THE CODE PROPERLY TO AVOID THIS LATER.
                global emb_array 
           
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

            

@app.task
def search(img):
    pass