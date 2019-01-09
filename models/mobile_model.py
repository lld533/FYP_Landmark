import os
import sys
sys.path.append(os.path.realpath('..'))
from tf_isqrt import isqrt_cov
import tensorflow as tf
from utils import preprocess


def model_mobile_v1(x):
    x = tf.layers.conv2d(x, 16, 18, padding='valid')
    x = tf.layers.max_pooling2d(x, 2, 1) #18x18x16 
    
    x = tf.layers.conv2d(x, 48, 8, padding='valid')
    x = tf.layers.max_pooling2d(x, 2, 1) #8x8x48
    
    x = tf.layers.conv2d(x, 64, 3, padding='valid')
    x = tf.layers.max_pooling2d(x, 2, 1) #3x3x64
    
    
    x = tf.layers.conv2d(x, 64, 2)
    x = tf.layers.flatten(x)
    
    
    # dense1
    dense1 = tf.layers.dense(
            x,
            1024,
            activation=None,
            use_bias=True,
            name='dense1')
    
    # regression
    # Dense layer 2, also known as the output layer.
    logits68 = tf.layers.dense(
        inputs=dense1,
        units=136,
        activation=None,
        use_bias=True,
        name="logits68")   
    
    return logits68

def model_mobile_gap(x):
    '''
    Substitute flatten with global average pooling
    '''
    x = preprocess(x)
    x = tf.layers.conv2d(x, 64, 5, padding='valid')
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)    
    x = tf.layers.max_pooling2d(x, 2, 2) #Nx18x18x64
    
    x = tf.layers.conv2d(x, 128, 3, padding='valid')
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)    
    x = tf.layers.max_pooling2d(x, 2, 2) #Nx8x8x128
    
    x = tf.layers.conv2d(x, 256, 3, padding='valid')
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, 2, 2) #Nx3x3x256
    
    
    x = tf.layers.conv2d(x, 512, 1, padding='same') #Nx3x3x512
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    
    x = tf.reduce_mean(x, axis=[1,2]) #Nx512
        
    # regression
    # the output layer.
    logits68 = tf.layers.dense(
        inputs=x,
        units=136,
        activation=None,
        use_bias=True,
        name="logits68")   
    
    return logits68

def model_mobile_isqrtcov(x):
    '''
    Substitute flatten with iSQRT-COV (global covariance pooling)
    '''
    x = preprocess(x)
    x = tf.layers.conv2d(x, 16, 5, padding='valid')
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)     
    x = tf.layers.max_pooling2d(x, 2, 2) #Nx18x18x16 
    
    x = tf.layers.conv2d(x, 48, 3, padding='valid')
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)     
    x = tf.layers.max_pooling2d(x, 2, 2) #Nx8x8x48
    
    x = tf.layers.conv2d(x, 64, 1, padding='same') # maybe an extra 1x1 conv layer will make it better
    
    x = isqrt_cov(x, T=3, data_format ='channels_last') #Nx[64*(64+1)/2]
    
    # regression
    # the output layer.
    logits68 = tf.layers.dense(
        inputs=x,
        units=136,
        activation=None,
        use_bias=True,
        name="logits68")   
    
    return logits68