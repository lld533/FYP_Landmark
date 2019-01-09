import numpy as np
import math
import tensorflow as tf

#https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
def gkern(l=45, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.sum(kernel)

def make_guassian_template():
    l = 45
    result = np.zeros((l, l))
    for x in range(1,11,2):
        result += x * x * gkern(l=l, sig=float(x))
    
    result *= 0.4 * math.pi
    result = np.expand_dims(np.expand_dims(result, -1), -1)
    return tf.convert_to_tensor(result, dtype=tf.float32)

def kernel_conv(imgs, template, data_format='channels_last'):
    if data_format == 'channels_last':
        imgs = tf.transpose(imgs, perm=[0,3,1,2])
    
    # https://stackoverflow.com/questions/35565312/is-there-a-convolution-function-in-tensorflow-to-apply-a-sobel-filter
    def kernel_conv_core(x):
        x = tf.expand_dims(tf.expand_dims(x, axis=0), axis=-1)
        return  tf.nn.conv2d(x, 
                             template,
                             strides=[1, 1, 1, 1], 
                             padding='SAME')        
                
    imgs_reshape = tf.reshape(imgs, [-1, int(imgs.shape[-2]), int(imgs.shape[-1])])
    imgs_filtered = tf.map_fn(lambda x: kernel_conv_core(x), imgs_reshape)
    
    imgs_filtered = tf.reshape(imgs_filtered, 
                               [-1, int(imgs.shape[1]), int(imgs.shape[2]), int(imgs.shape[3])])
    
    if data_format == 'channels_last':
        imgs_filtered = tf.transpose(imgs_filtered, 
                                     perm=[0,2,3,1])
        
    return imgs_filtered
    
def model_desktop_gap(x, data_format='channels_last'):
    template = make_guassian_template()
    
    for i in range(5):
        x = tf.layers.conv2d(x, 64, 5, padding='SAME')
    for i in range(10):
        x = tf.layers.conv2d(x, 128, 3, padding='SAME')
    
    
    x_l = tf.layers.conv2d(x, 68, 1)
    x_l = kernel_conv(x_l, template)
    
    x_r = tf.layers.conv2d(x, 68, 1)
    x_r = kernel_conv(x_r, template)
    
    x = tf.concat([x_l, x_r], axis=-1)
    
    for i in range(7):
        x = tf.layers.conv2d(x, 128, 3, padding='SAME', dilation_rate=(4,4))
    
    x = tf.layers.conv2d(x, 68, 1)
    
    return x

# https://stackoverflow.com/questions/36388431/tensorflow-multi-dimension-argmax
def argmax_2d(tensor):
    # flatten the Tensor along the height and width axes
    flat_tensor = tf.reshape(tensor, 
                             (-1, int(tensor.shape[1]*tensor.shape[2]), tensor.shape[3]))

    # argmax of the flat tensor
    argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)

    # convert indexes into 2D coordinates
    argmax_x = argmax // tf.shape(tensor)[2]
    argmax_y = argmax % tf.shape(tensor)[2]

    # concat and return coordinates
    return tf.concat([argmax_x, argmax_y], axis=-1)

def heatmap2coords(heatmaps):
    # Input:
    # heatmap: [N, H, W, K], N: batch size, H: height, W: width, K: #keypoints
    #
    # Output
    # coords: [N, 2K], N: batch size, K: #keypoints organized by x_1..x_Ky_1..y_K
    return tf.cast(argmax_2d(heatmaps), dtype=tf.float32)

"""
Test code
"""
## function model_desktop_gap
#imgs = tf.placeholder(tf.float32, [None, 96,96, 3])
#prediction_heatmap = model_desktop_gap(imgs)

## function heatmap2coords
#np_heatmap = np.random.rand(1,96,96,68)
#pseudo_heatmap = tf.convert_to_tensor(np_heatmap, dtype=tf.float32)
#prediction_coords = heatmap2coords(pseudo_heatmap)

#print("="*32)
#print("np_heatmap:")
#print(np_heatmap)
#print("="*32)
#print("coords:")

#with tf.Session() as sess:
    #print(sess.run(prediction_coords))