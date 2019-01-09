import tensorflow as tf
import os
import sys
sys.path.append(os.path.realpath('..'))
from tf_isqrt import isqrt_cov
from utils import grid_to_heatmaps, grid_restore_coords

def model_mobile_cls_reg_gap(x, data_format='channels_last'):
    '''
    Substitute flatten with classification part and regression part
    '''
    x = tf.layers.conv2d(x, 16, 5, padding='valid')
    x = tf.layers.max_pooling2d(x, 2, 2) #Nx18x18x16 
    
    x = tf.layers.conv2d(x, 68, 3, padding='valid')
    x = tf.layers.max_pooling2d(x, 2, 2) #Nx8x8x68
    
    # TODO @_@: to support NCHW
    #NHWC 
    prj_x = tf.reduce_sum(x, axis=2) #Nx8x68
    prj_y = tf.reduce_sum(x, axis=1) #Nx8x68
    
    prj_x = tf.layers.conv1d(prj_x, 
                             int(prj_x.shape[-1]), 
                             3, 
                             padding='same')
    prj_y = tf.layers.conv1d(prj_y, 
                             int(prj_y.shape[-1]), 
                             3, 
                             padding='same')
    
    part_cls = tf.concat([prj_x, prj_y], -1) #Nx8x136
    part_cls = tf.transpose(part_cls, perm=[0,2,1]) #Nx136x8
    
    attn_maps = grid_to_heatmaps(part_cls) #Nx68x8x8
    attn_maps = tf.transpose(attn_maps, perm=[0,2,3,1]) #Nx8x8x68
    
    # combine attn_maps & x
    x = tf.concat([x, attn_maps], -1) #Nx8x8x136
    
    x = tf.layers.conv2d(x, 256, 3, padding='valid')
    x = tf.layers.max_pooling2d(x, 2, 2) #Nx3x3x256
        
    x = tf.reduce_mean(x, axis=[1,2]) #Nx256
        
    # regression
    # the output layer.
    part_reg = tf.layers.dense(
                inputs=x,
                units=136,
                activation=None,
                use_bias=True)
    
    logits68 = grid_restore_coords(part_cls, 
                                   part_reg, 
                                   40, 
                                   N=8, 
                                   padded_N=0)
    
    return logits68

def model_mobile_cls_reg_isqrtcov(x, data_format='channels_last'):
    '''
    Substitute flatten with classification part and regression part
    '''
    x = tf.layers.conv2d(x, 16, 5, padding='valid')
    x = tf.layers.max_pooling2d(x, 2, 2) #Nx18x18x16 
    
    x = tf.layers.conv2d(x, 68, 3, padding='valid')
    x = tf.layers.max_pooling2d(x, 2, 2) #Nx8x8x68
    
    # TODO @_@: to support NCHW
    #NHWC 
    prj_x = tf.reduce_sum(x, axis=2) #Nx8x68
    prj_y = tf.reduce_sum(x, axis=1) #Nx8x68
    
    prj_x = tf.layers.conv1d(prj_x, 
                             int(prj_x.shape[-1]), 
                             3, 
                             padding='same')
    prj_x = tf.layers.batch_normalization(prj_x)
    prj_x = tf.nn.relu(prj_x)
    prj_y = tf.layers.conv1d(prj_y, 
                             int(prj_y.shape[-1]), 
                             3, 
                             padding='same')
    prj_y = tf.layers.batch_normalization(prj_y)
    prj_y = tf.nn.relu(prj_y)
    
    part_cls = tf.concat([prj_x, prj_y], -1) #Nx8x136
    part_cls = tf.transpose(part_cls, perm=[0,2,1]) #Nx136x8
    
    attn_maps = grid_to_heatmaps(part_cls) #Nx68x8x8
    attn_maps = tf.transpose(attn_maps, perm=[0,2,3,1]) #Nx8x8x68
    
    # combine attn_maps & x
    x = tf.concat([x, attn_maps], -1) #Nx8x8x136
    
    x = tf.layers.conv2d(x, 256, 3, padding='same') #Nx8x8x256
    x = tf.layers.conv2d(x, 64, 1) #Nx8x8x64
    #x = tf.layers.max_pooling2d(x, 2, 2) #Nx3x3x64
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    
    x = isqrt_cov(x) #Nx[64x(64+1)/2]
        
    # regression
    # the output layer.
    part_reg = tf.layers.dense(
                inputs=x,
                units=136,
                activation=None,
                use_bias=True)
    
    logits68 = grid_restore_coords(part_cls, 
                                   part_reg, 
                                   40, 
                                   N=8, 
                                   padded_N=0)
    
    return logits68


# FOR TEST
#x = tf.placeholder(tf.float32, [None, 40, 40, 3])
#logits68 = model_mobile_cls_reg_gap(x)
#logits68 = model_mobile_cls_reg_isqrtcov(x)