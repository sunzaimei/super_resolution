import os
import time

import cv2
import numpy as np
import tensorflow as tf

DATASET_DIR = "/data/DIV2K/"
DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    start_time = time.perf_counter()
    lr_batch = tf.cast(lr_batch, tf.float32)
    # model_fn = tf.function(model)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    runtime = (time.perf_counter()-start_time)*1000
    return sr_batch, runtime


def evaluate(model, dataset, save_result=False, checkpoint_dir=''):
    if save_result:
        save_rediction_dir = os.path.join(checkpoint_dir, f"images_result")
        save_gt_dir = os.path.join(checkpoint_dir, f"images_gt_hr")
        if not os.path.exists(save_rediction_dir):
            os.makedirs(save_rediction_dir)
        if not os.path.exists(save_gt_dir):
            os.makedirs(save_gt_dir)
    psnr_values = []
    runtime_values = []
    count = 800
    for lr, hr in dataset:
        sr, runtime = resolve(model, lr)
        # print("image runtime", runtime)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
        runtime_values.append(runtime)
        if save_result:
            image = sr.numpy()[0]
            count += 1
            cv2.imwrite(os.path.join(save_rediction_dir, f"{str(count)}.png"),
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_gt_dir, f"{str(count)}.png"),
                        cv2.cvtColor(hr.numpy()[0], cv2.COLOR_RGB2BGR))
    avg_psnr = tf.reduce_mean(psnr_values)
    avg_runtime = tf.reduce_mean(runtime_values[10:])
    return avg_psnr, score(avg_psnr, avg_runtime), avg_runtime


# ---------------------------------------
#  Normalization
# ---------------------------------------


def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)


def score(psnr, runtime):
    """
    :param psnr:
    :param runtime: unit is in ms
    :return:
    """
    return tf.math.pow(2, 2*psnr) / (2.81475e14 * tf.convert_to_tensor(runtime, tf.float32))

# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)