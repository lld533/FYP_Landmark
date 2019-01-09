import tensorflow as tf

def preprocess(imgs, gradients=True, dct=False, data_type="channel_last"):
    """
    PREPROCESS convert an BGR image to 
    YUV-Gradients(optional)-DCT(optional)
    """
    arr = []
    yuv = tf.image.rgb_to_yuv(imgs)
    arr.append(yuv)
    
    if gradients:
        dy, dx = tf.image.image_gradients(imgs)
        val = tf.sqrt(dy * dy + dx * dx)
        arr.append(val)
    
    if dct:
        if data_type == "channel_last":
            # DCT works on last channel, so for "channel_last"
            # 1/ convert channel dim from -1 to 1 [0,3,1,2]
            # 2/ run DCT and swap last two dims [0,3,2,1]
            # 3/ run DCT again [0,3,2,1]
            # 4/ swap back to [0,1,2,3]            
            val = tf.transpose(
                tf.spectral.dct(
                    tf.transpose(
                        tf.spectral.dct(
                            tf.transpose(imgs, 
                                         perm=[0,3,1,2]),
                                norm="ortho"), # step 1/
                            perm=[0,1,3,2]), # step 2/
                        norm="ortho"
                    ), # step 3/
                perm=[0,3,2,1]) # step 4/
        else:
            # DCT works on last channel, so for "channel_first"
            # 1/ run DCT [0,1,2,3]
            # 2/ swap last two dims [0,1,3,2]
            # 3/ run DCT again [0,1,3,2]
            # 4/ swap back to [0,1,2,3]             
            val = tf.transpose(
                tf.spectral.dct(
                    tf.transpose(
                        tf.spectral.dct(imgs,
                                        norm="ortho"), # step 1/
                            perm=[0,1,3,2]), # step 2/
                        norm="ortho"
                    ), # step 3
                perm=[0,1,3,2]) # step 4
        
        arr.append(val)

    if data_type == "channel_last":        
        result = tf.concat(arr, axis=-1)
    elif data_type == "channel_first":
        result = tf.concat(arr, axis=1)    
    return result

def grid_split(labels, a, N=8, padded_N=2):
    """
    GRID_SPLIT convert landmarks from [xx...xxyy...yy]
    into one-hot classes and corresponding position to
    the center of the assigned local grid.
    
    For example,
    A point [3.2,7.1] in a 40x40 image will be converted to
    classes=[[0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0]]
    position=[0.7,-0.4],
    as 0*5+2.5+0.7=3.2; 1*5+2.5-0.4=7.1
    
    Note:
    A valid value is expected to be located in [-padded_N, N+padded_N-1].
    Those out-of-the-box values will be assigned to the closest grid.
    
    Input:
    labels: [batch_size, 2K], values are not normalized.
    a: edge length of the image.
    N: number grids along width/height.
    
    Output:
    classes: [batch_size, 2K, N+2*padded_N] one hot class labels
    position: [batch_size, 2K] position in the assigned
            local grid.
    """
    grid_sz = a / N
    half_grid_sz = grid_sz / 2.0
    val = labels / grid_sz
    
    floor_val = tf.floor(val)
    floor_val = tf.clip_by_value(floor_val, 
                                 -padded_N, 
                                 padded_N + N - 1)
    
    position = labels - (floor_val + 1) * grid_sz + half_grid_sz
    classes = tf.one_hot(tf.cast(floor_val + padded_N, tf.int32), 
                         int(N + padded_N * 2))
    
    return classes, position

def grid_restore_coords(classes, position, a, N=8, padded_N=2):
    """
    GRID_SPLIT convert landmarks from [xx...xxyy...yy]
    into one-hot classes and corresponding position to
    the center of the assigned local grid.
    
    For example,
    A point [3.2,7.1] in a 40x40 image will be restored from
    classes=[[0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0]]
    position=[0.7,-0.4],
    as 0*5+2.5+0.7=3.2; 1*5+2.5-0.4=7.1
    
    Note:
    A valid value is expected to be located in [-padded_N, N+padded_N-1].
    Those out-of-the-box values will be assigned to the closest grid.
        
    Input:
    classes: [batch_size, 2K, N+2*padded_N] one hot class labels
    position: [batch_size, 2K] position in the assigned
            local grid.
    a: edge length of the image.
    N: number grids along width/height.
    
    Output:
    labels: [batch_size, 2K], values are not normalized.
    """    
    grid_sz = a / N
    half_grid_sz = grid_sz / 2.0
    
    cls = tf.argmax(classes, axis=-1, output_type=tf.int32)
    cls = cls - padded_N
    return tf.cast(cls, dtype=tf.float32) * grid_sz + position + half_grid_sz

def grid_to_heatmaps(classes):
    """
    GRID_TO_HEATMAPS renders heatmaps for each point based on 
    their possibilities of each classes along x&y axis. 
    See defination in grid_split.
    
    Note:
    A valid value is expected to be located in [-padded_N, N+padded_N-1].
    Those out-of-the-box values will be assigned to the closest grid.
        
    Input:
    classes: [batch_size, 2K, N+2*padded_N] one hot class labels
    
    Output:
    heatmaps: [batch_size, K, N+2*padded_N, N+2*padded_N]
    """      
    cls = tf.reshape(classes, [-1, 2, int(classes.shape[1]) // 2, int(classes.shape[2])])
    cls_x = cls[:,0,:,:]
    cls_y = cls[:,1,:,:]
    
    cls_x = tf.reshape(cls_x, [-1, int(classes.shape[2])])
    cls_y = tf.reshape(cls_y, [-1, int(classes.shape[2])])
    
    def norm_cls(cls):
        result = tf.clip_by_value(cls, 0.0, 1.0)
        s = tf.reduce_sum(result) + 1e-6        
        return result / s
    
    def cls2map(cls_x, cls_y):
        prob_x, prob_y = tf.meshgrid(norm_cls(cls_x), 
                                     norm_cls(cls_y))
        return  prob_x * prob_y, tf.zeros_like(prob_x)
    
    heatmaps = tf.map_fn(lambda x: cls2map(x[0], x[1]), 
                        (cls_x, cls_y), 
                        dtype=(cls_x.dtype, cls_y.dtype))
    
    heatmaps = heatmaps[0]
    
    heatmaps = tf.reshape(heatmaps, 
                          [-1, int(classes.shape[1]) // 2, int(classes.shape[2]), int(classes.shape[2])])
    
    return heatmaps

"""
Test of grid splittion
"""
import numpy as np
N = 8
padded_N = 2
a = 40
coords = np.asarray([[3.2, 30.2, 40.2, 7.1, 25.6, -3.3]])
coords = tf.convert_to_tensor(coords, dtype=tf.float32)
cls, pos = grid_split(coords, a, N=N, padded_N=padded_N)
#coords_restored = grid_restore_coords(cls, pos, a, N=N, padded_N=padded_N)

#cls = cls + tf.random.uniform(cls.shape, -0.15, 0.15)

heatmaps = grid_to_heatmaps(cls)

with tf.Session() as sess:
    #print(sess.run(coords_restored))
    print(sess.run(heatmaps))
    
"""
Test of YUV & gradients
"""
#import numpy as np
#import cv2
#a = cv2.imread("D:/test.jpg")
#a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
#a = cv2.resize(a, (40, 40))
#a = np.expand_dims(a, axis=0)
#a = tf.convert_to_tensor(a, dtype=tf.float32)
#b = preprocess(a, gradients=True, dct=False)
#with tf.Session() as sess:
    #b_np = sess.run(b)
    #print(np.max(b_np.flat, axis=0))
    