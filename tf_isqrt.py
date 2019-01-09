# TF implementation of Peihua LI's CVPR 2018
import tensorflow as tf #tf.__version__>=1.12
import math

def eyes_like(mat, name=None):
    return tf.linalg.set_diag(tf.zeros_like(mat), tf.ones_like(tf.linalg.diag_part(mat)), name=name)

def covariance_mat(net, data_format ='channels_last'):   
    if data_format == 'channels_last':
        #net shape: [?,d,C]        
        eye_mat = eyes_like(tf.matmul(net, net, transpose_b=True))
        val = tf.constant(int(net.shape[1]), dtype=eye_mat.dtype)
        eye_bar = (eye_mat - tf.ones_like(eye_mat) / val) / val
        cov_mat = tf.matmul(tf.matmul(tf.transpose(net, 
                                                   perm=[0,2,1]), 
                                      eye_bar),
                            net)
    elif data_format == 'channels_first':
        #net shape: [?,C,d]
        eye_mat = eyes_like(tf.matmul(net, net, transpose_a=True))
        val = tf.constant(int(net.shape[2]), dtype=eye_mat.dtype)
        eye_bar = (eye_mat - tf.ones_like(eye_mat) / val) / val 
        cov_mat = tf.matmul(tf.matmul(net,
                                      eye_bar),
                            tf.transpose(net,
                                         perm=[0,2,1])) 
    return cov_mat #[?,C,C]    

# method == 'fro' or 'trace'
def pre_norm(cov_mat, method='trace'):
    if method == 'fro':
        cov_mat_norm = tf.norm(cov_mat, ord='fro', axis=[-2,-1], keep_dims=True)
    elif method == 'trace':
        cov_mat_norm = tf.trace(cov_mat)
        cov_mat_norm = tf.expand_dims(tf.expand_dims(cov_mat_norm, axis=-1), axis=-1)
            
    cov_mat_norm = tf.tile(cov_mat_norm, [1,cov_mat.shape[-2], cov_mat.shape[-1]])
    Y = tf.divide(cov_mat, cov_mat_norm, name='nw_y0')
    Z = eyes_like(Y, name='nw_z0')    
    
    return cov_mat_norm, Y, Z


def isqrt_NS(cov_mat, Y, Z, T):
    for i in range(T):
        aux_mat = 1.5 * eyes_like(Y) - 0.5 * tf.matmul(Z, Y)
        Y = tf.matmul(Y, aux_mat, name='nw_y%d'%(i+1))
        Z = tf.matmul(aux_mat, Z, name='nw_z%d'%(i+1))
    
    return Y

def post_compensate(Y, cov_mat_norm):
    return tf.multiply(tf.sqrt(cov_mat_norm), Y)

def flatten_and_extract_triangular_matrix(C):
    ones = tf.ones_like(C, dtype=tf.bool)

    d = int(C.shape[-1])
    output_dim = int(d * (d + 1) / 2)

    mask = tf.matrix_band_part(ones, 0, -1)
    result = tf.boolean_mask(C, mask)
    
    return tf.reshape(result, [-1, output_dim])

def isqrt_cov(input_feat, T=3, data_format ='channels_last'):
    """
    Tensorflow implementation of iSQRT-COV proposed by LI et al. in CVPR2018
    input_feat: input features, 4-D tensor.
    T: number of iteration, usually 3 or 5 (default: 3)
    data_format: 'channels_first' or 'channels_last' (default)
    """
    input_shape = input_feat.shape

    if len(input_shape) != 4:
        return input_feat
    
        
    if data_format == 'channels_last':
        # reshape wxhxd to nxd, where n = wxh
        net = tf.reshape(input_feat, [-1,
                                      int(input_shape[1]) * int(input_shape[2]),
                                      int(input_shape[-1])])
    elif data_format == 'channels_first':
        # reshape wxhxd to nxd, where n = wxh
        net = tf.reshape(input_feat, [-1,
                                      int(input_shape[1]),
                                      int(input_shape[2]) * int(input_shape[3])])   
    
    # covariance matrix
    cov_mat = covariance_mat(net, data_format=data_format)
    
    # pre-normalization
    cov_mat_norm, Y, Z = pre_norm(cov_mat)
    
    # Newton-Schulz iteration
    Y = isqrt_NS(cov_mat, Y, Z, T)
    
    # Post compensation
    C = post_compensate(Y, cov_mat_norm)
    
    C = flatten_and_extract_triangular_matrix(C)
    return C  

#import numpy as np
#val = np.random.rand(2,3,3,32)
#x = tf.convert_to_tensor(val)

#x = isqrt_cov(x)

#with tf.Session() as sess:
    #print(sess.run(x))
