import os
import sys
sys.path.append(os.path.realpath('./'))
from object_detection.core.preprocessor import random_horizontal_flip, random_vertical_flip, random_rotation90, random_adjust_brightness, random_rgb_to_gray, random_adjust_contrast, random_adjust_hue, random_adjust_saturation

import numpy as np
import tensorflow as tf
import math
import cv2
from models.mobile_model import model_mobile_gap

IMG_WIDTH = 40
IMG_HEIGHT = 40
IMG_CHANNEL = 3

tf.logging.set_verbosity(tf.logging.INFO)

data_dir = "C:/Seagate backup/DJI_Landmark_Nov/10K_tfrecords_40"
model_dir = "D:/train/"
test_img_dir = "C:/Seagate backup/DJI_Landmark_Nov/10K_release/test"
test_output_dir = './output_dir/'
saved_model_dir = '/saved_model/'


LDMK68_HORIZONTAL_PERMUTATION = [16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,26,25,24,23,22,21,20,19,18,17,27,28,29,30,35,34,33,32,31,45,44,43,42,47,46,39,38,37,36,41,40,54,53,52,51,50,49,48,59,58,57,56,55,64,63,62,61,60,67,66,65]
LDMK83_HORIZONTAL_PERMUTATION = [0,10,11,12,13,14,15,16,17,18,1,2,3,4,5,6,7,8,9,65,66,71,69,68,70,67,72,74,73,79,78,77,76,75,82,81,80,46,38,42,43,44,39,40,41,45,37,47,51,52,53,48,49,50,54,59,60,61,58,55,56,57,63,62,64,19,20,25,23,22,24,21,26,28,27,33,32,31,30,29,36,35,34]
LDMK68_VERTICAL_PERMUTATION =   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,26,25,24,23,22,21,20,19,18,17,27,28,29,30,35,34,33,32,31,45,44,43,42,47,46,39,38,37,36,41,40,54,53,52,51,50,49,48,59,58,57,56,55,64,63,62,61,60,67,66,65]
LDMK83_VERTICAL_PERMUTATION =   list(range(83))

def cnn_model_fn_mobile(features, labels, mode):
    x = tf.to_float(features['x'], name='input_to_float')
    x = tf.image.resize_images(x, [40,40])

    logits68 = model_mobile_gap(x)
    
    # Make prediction for PREDICATION mode.
    predictions_dict = {
        "name": features['name'],
        "logits68": logits68,
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict)

    # Caculate loss using mean squared error.
    label_tensor68 = tf.convert_to_tensor(labels["labels68"], dtype=tf.float32)
    loss68 = tf.losses.mean_squared_error(
        labels=label_tensor68, predictions=logits68)

    # Configure the train OP for TRAIN mode.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss68, 
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss68,
            train_op=train_op,
            export_outputs={'marks68': tf.estimator.export.RegressionOutput(logits68)})

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "MSE68": tf.metrics.root_mean_squared_error(
            labels=label_tensor68,
            predictions=logits68),
        }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss68, eval_metric_ops=eval_metric_ops)    
    
cnn_model_fn = cnn_model_fn_mobile

def _parse_function(record):
    """
    Extract data from a `tf.Example` protocol buffer.
    """
    # Defaults are not specified since both keys are required.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string),        
        'image/height': tf.FixedLenFeature([1], dtype=tf.int64),
        'image/width': tf.FixedLenFeature([1], dtype=tf.int64),        
        'image/object/landmark68': tf.FixedLenFeature([136], dtype=tf.float32),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value='')
    }      
    parsed_features = tf.parse_single_example(record, keys_to_features)
    
    width = tf.cast(parsed_features['image/width'], tf.int32)
    height = tf.cast(parsed_features['image/height'], tf.int32)
    
    # Extract features from single example
    image_decoded = tf.image.decode_jpeg(parsed_features['image/encoded'])
    image_reshaped = tf.reshape(
        image_decoded, 
        ( IMG_HEIGHT, IMG_WIDTH,IMG_CHANNEL))
    image_reshaped = tf.cast(image_reshaped, tf.float32)
    
    landmark68 = tf.cast(parsed_features['image/object/landmark68'], tf.float32)

    return {"x": image_reshaped, "name": parsed_features['image/filename']}, {"labels68":landmark68}
    
    
def normalize_labels(labels, scale=40.0):
    # NORMALIZE_LABELS makes each element in labels to [0.0,1.0]
    # via dividing each element by the scale value.
    return labels / scale

def rescale_labels(labels, scale=40.0):
    # RESCALE_LABELS restore normalized labels to given scale
    # via multiplying the scale value.    
    return labels * scale

def input_fn(record_file, batch_size, num_epochs=None, shuffle=True, data_augmentation=False):
    """
    Input function required for TensorFlow Estimator.
    """
    dataset = tf.data.TFRecordDataset(record_file)

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(_parse_function)
    
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=10000)
                
    if batch_size != 1:
        dataset = dataset.batch(batch_size)
    if num_epochs != 1:
        dataset = dataset.repeat(num_epochs)

    # Make dataset iteratable.
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    feature, labels = iterator.get_next()
    
    labels68 = labels['labels68']
    
    if data_augmentation:
        # In prior to data agumentation, labels should be normalized and
        # reshaped to [num_sample, num_point, 2].
        labels68 = tf.reshape(labels68, [-1, 2, int(labels68.shape[-1]) // 2])
        labels68 = tf.transpose(labels68, perm=[0,2,1])        
        labels68 = normalize_labels(labels68)
        
        def data_augmentation_for_training(img, label):
            # from x-y to y-x
            label = label[:,::-1]            
            label = tf.expand_dims(label, 0)
            
            
            img, label = random_horizontal_flip(img, 
                                                keypoints=label,
                                                keypoint_flip_permutation=LDMK68_HORIZONTAL_PERMUTATION)
            # Vertical flip brings no good..
            #img, label = random_vertical_flip(img, 
                                              #keypoints=label,
                                              #keypoint_flip_permutation=LDMK68_VERTICAL_PERMUTATION)
            img, label = random_rotation90(img, keypoints=label)
            
            img = random_adjust_brightness(img)
            img = random_rgb_to_gray(img) 
            img = random_adjust_contrast(img)
            img = random_adjust_hue(img)
            img = random_adjust_saturation(img)
            
            label = tf.squeeze(label)
            
            # from y-x back to x-y
            label = label[:,::-1]            
            
            return img, label
        
        imgs = feature['x']
        
        c = tf.map_fn(lambda x: data_augmentation_for_training(x[0], x[1]), 
                      (imgs, labels68), 
                      dtype=(imgs.dtype, labels68.dtype))
        
        feature['x'] = c[0]
        labels68 = c[1]
        
        
        # After data agumentation, labels should be rescaled and
        # reshaped to [num_sample, num_point * 2].        
        labels68 = rescale_labels(labels68)
        labels68 = tf.transpose(labels68, perm=[0,2,1])
        labels68 = tf.reshape(labels68, [-1, labels68.shape[1] * labels68.shape[2]])
        labels['labels68'] = labels68
        
    
    return feature, labels


def _train_input_fn():
    """Function for training."""
    record_file = os.path.join(data_dir, "train.tfrecords")
    return input_fn(
        record_file=record_file, 
        batch_size=32, 
        num_epochs=50, 
        shuffle=True,
        data_augmentation=True)


def _eval_input_fn():
    """Function for evaluating."""
    record_file = os.path.join(data_dir, "validation.tfrecords")
    return input_fn(
        record_file=record_file,
        batch_size=2,
        num_epochs=1,
        shuffle=False,
        data_augmentation=False)


def _predict_input_fn():
    """Function for predicting."""
    record_file = os.path.join(data_dir, "test.tfrecords")
    return input_fn(
        record_file=record_file,
        batch_size=2,
        num_epochs=1,
        shuffle=False,
        data_augmentation=False)


def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.Example."""
    image = tf.placeholder(dtype=tf.float32,
                           shape=[None, IMG_HEIGHT,IMG_WIDTH, IMG_CHANNEL],
                           name='input_image_tensor')
    receiver_tensor = {'x': image, 
                       'name': tf.placeholder(tf.string, shape=[None]), 
                       'labels68': tf.placeholder(tf.float32, shape=[None, 68*2])}
    feature = tf.reshape(image, [-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
    return tf.estimator.export.ServingInputReceiver(receiver_tensor, receiver_tensor)



    
def main(unused_argv):
    """MAIN"""
    # Create the Estimator
    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir)

    # Choose mode between Train, Evaluate and Predict
    mode_dict = {
        'train': tf.estimator.ModeKeys.TRAIN,
        'eval': tf.estimator.ModeKeys.EVAL,
        'predict': tf.estimator.ModeKeys.PREDICT
    }

    
    #The following two lines are used for training
    for i in range(200):
        mode = mode_dict['train'] if i % 2 == 0 else mode_dict['eval']
        
    
    #While the following two lines are used for prediction.    
    #for i in range(1):
        #mode = mode_dict['predict']
    
        if mode == tf.estimator.ModeKeys.TRAIN:
            estimator.train(input_fn=_train_input_fn, steps=200000)
            
            # Export result as SavedModel.
            estimator.export_savedmodel(saved_model_dir, serving_input_receiver_fn)
    
        elif mode == tf.estimator.ModeKeys.EVAL:
            evaluation = estimator.evaluate(input_fn=_eval_input_fn)
            print(evaluation)
    
        else:
            predictions = estimator.predict(input_fn=_predict_input_fn)
            scale = 40.0
            for _, result in enumerate(predictions):
                img = cv2.imread(os.path.join(test_img_dir, result['name'].decode('ASCII')))
                marks68 = np.reshape(result['logits68'], (2,-1))
                marks68 = np.transpose(marks68)
                
                for mark in marks68:
                    cv2.circle(img, 
                               (int(round(mark[0] / scale * img.shape[1])), int(round(mark[1] / scale * img.shape[1]))), 
                               1, 
                               (0, 255, 0), 
                               -1, 
                               cv2.LINE_AA)

                temp_path=result['name'].decode('ASCII').split('\\')[0]
                
                if not os.path.exists(os.path.join(test_output_dir, temp_path)):
                    os.makedirs(os.path.join(test_output_dir, temp_path))
                cv2.imwrite(os.path.join(test_output_dir, result['name'].decode('ASCII')), img)


if __name__ == '__main__':
    tf.app.run()
