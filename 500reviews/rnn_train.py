#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 00:28:59 2018

@author: Di
"""

import numpy as np
import tensorflow as tf
import pickle
import time
import os
import collections
from simple_model import Model
#from model import Model

data_dir = '/Users/Di/Documents/2017Fall/NLG/rnn'
input_encoding = None # character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings'
log_dir = '/Users/Di/Documents/2017Fall/NLG/rnn/log'# directory containing tensorboard logs
save_dir = '/Users/Di/Documents/2017Fall/NLG/rnn/out' # directory to store checkpointed models
rnn_size = 256 # size of RNN hidden state
num_layers = 2 # number of layers in the RNN
model = 'lstm' # lstm model
batch_size = 50 # minibatch size
seq_length = 25 # RNN sequence length
num_epochs = 25 # number of epochs
save_every = 1000 # save frequency
grad_clip = 5. #clip gradients at this value
learning_rate= 0.002 #learning rate
decay_rate = 0.97 #decay rate for rmsprop
gpu_mem = 0.666 #%% of gpu memory to be allocated to this process. Default is 66.6%%
init_from = None



input_file = os.path.join(data_dir, "reviews.txt")
vocab_file = os.path.join(data_dir, "vocab.pkl")
tensor_file = os.path.join(data_dir, "data.npy")

with open(input_file, "r") as f:
    data = f.read()
    
x_text = data.split()




# count the number of words
word_counts = collections.Counter(x_text)

# Mapping from index to word : that's the vocabulary
vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))

# Mapping from word to index
vocab = {x: i for i, x in enumerate(vocabulary_inv)}
words = [x[0] for x in word_counts.most_common()]
vocab_size = len(words)  

with open(vocab_file, 'wb') as f:
    pickle.dump(words, f, protocol=pickle.HIGHEST_PROTOCOL)

    
    
tensor = np.array(list(map(vocab.get, x_text)))

# Save the data to data.npy
np.save(tensor_file, tensor)

print('tensor is:' + str(tensor))
print("It's shape: " + str(np.shape(tensor)))


num_batches = int(tensor.size / (batch_size * seq_length))
print('number of batches is: ' + str(num_batches))

tensor = tensor[:num_batches * batch_size * seq_length]
print('The shape of the new tensor is: '+ str(np.shape(tensor)))

xdata = tensor
ydata = np.copy(tensor)

ydata[:-1] = xdata[1:]
ydata[-1] = xdata[0]

x_batches = np.split(xdata.reshape(batch_size, -1), num_batches, 1)
y_batches = np.split(ydata.reshape(batch_size, -1), num_batches, 1)


pointer = 0
with open(os.path.join(save_dir, 'words_vocab.pkl'), 'wb') as f:
    pickle.dump((words, vocab), f, protocol=pickle.HIGHEST_PROTOCOL)
    
model = Model(data_dir,input_encoding,log_dir,save_dir,rnn_size,num_layers,model,batch_size,seq_length,num_epochs,save_every,grad_clip,learning_rate,decay_rate,gpu_mem,init_from, vocab_size)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem)





with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #add the session graph to the writer
        train_writer.add_graph(sess.graph)

        #initialize global variables
        tf.global_variables_initializer().run()

        #create the Saver to save the model and its variables.
        saver = tf.train.Saver(tf.global_variables())

        #create a for loop, to run over all epochs (defined as e)
        for e in range(model.epoch_pointer.eval(), num_epochs):
            #a session encapsulates the environement in which operations objects are executed.
                        
            #Initialization:
            
            #here we assign to the lr (learning rate) value of the model, the value : args.learning_rate * (args.decay_rate ** e))
            sess.run(tf.assign(model.lr, learning_rate * (decay_rate ** e)))
            
            #we define the state of the model. At the beginning, its the initial state of the model.
            state = sess.run(model.initial_state)
            #speed to 0 at the beginning.
            speed = 0
            #reinitialize pointer for batches
            pointer = 0
            
            if init_from is None:
                assign_op = model.epoch_pointer.assign(e)
                sess.run(assign_op)

            if init_from is not None:
                pointer = model.batch_pointer.eval()
                init_from = None

            #in each epoch, for loop to run over each batch (b)
            for b in range(pointer, num_batches):
                #define the starting date:
                start = time.time()
                #define x and y for the next batch
                x, y = x_batches[pointer], y_batches[pointer]
                pointer += 1

                #create the feeding string for the model.
                #input data are x, targets are y, the initiate state is state, and batch time 0.
                feed = {model.input_data: x, model.targets: y, model.initial_state: state,
                        model.batch_time: speed}

                #run the session and train.
                summary, train_loss, state, _, _ = sess.run([merged, model.cost, model.final_state,
                                                             model.train_op, model.inc_batch_pointer_op], feed)
                #add summary to the log
                train_writer.add_summary(summary, e * num_batches + b)

                #calculate the speed of the batch.
                #this information will be displayed later.
                speed = time.time() - start

                #display something in the console
                #---------------------------------
                #print information:
                if (e * num_batches + b) % batch_size == 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                        .format(e * num_batches + b,
                                num_epochs * num_batches,
                                e, train_loss, speed))
                
                #save model:
                if (e * num_batches + b) % save_every == 0 \
                        or (e==num_epochs-1 and b == num_batches-1): # save for the last result
                    #define the path to the model
                    checkpoint_path = os.path.join(save_dir, 'model_test.ckpt')
                    #save the model, woth increment ()
                    saver.save(sess, checkpoint_path, global_step = e * num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
        
        #close the session
        train_writer.close()
