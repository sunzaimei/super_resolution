import math
import os
import time
import cv2
import numpy as np
import tensorflow as tf

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def convert_2d(r, sigma=0.1):
    # tf.print(tf.shape(r))
    s = r + tf.random.normal(tf.shape(r), 0, sigma)
    # if np.min(s) >= 0 and np.max(s) <= 1:
    #     return s
    # # s = s - np.full(s.shape, np.min(s))
    # s = s * 1 / np.max(s)
    return s


def convert_3d(r, sigma=0.1):
    s_dsplit = []
    r = normalize(tf.cast(r, tf.float32))
    for d in range(r.shape[2]):
        rr = r[:, :, d]
        ss = convert_2d(rr, sigma=sigma)
        s_dsplit.append(ss)
    s = tf.stack(s_dsplit, axis=2)
    # s = np.dstack(s_dsplit)
    s = denormalize(s)
    tf.cast(s, tf.uint8)
    return s


def resolve(model, lr_batch):
    start_time = time.perf_counter()
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    runtime = (time.perf_counter()-start_time)*1000
    return sr_batch, runtime


def evaluate(model, dataset, save_result=False, checkpoint_dir=''):
    if save_result:
        assert os.path.exists(checkpoint_dir), "checkpoint_dir does not exist"
        save_result_dir = os.path.join(checkpoint_dir, f"images_result")
        save_gt_dir = os.path.join(checkpoint_dir, f"images_gt_hr")
        if not os.path.exists(save_result_dir):
            os.makedirs(save_result_dir)
        if not os.path.exists(save_gt_dir):
            os.makedirs(save_gt_dir)
    psnr_values = []
    runtime_values = []
    for lr, hr, name in dataset:
        filename = os.path.basename(name[0].numpy().decode('utf-8'))
        sr, runtime = resolve(model, lr)
        psnr_value = np.average(psnr(hr, sr).numpy())
        psnr_values.append(psnr_value)
        runtime_values.append(runtime)
        if save_result:
            image = sr.numpy()[0]
            cv2.imwrite(os.path.join(save_result_dir, filename),
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_gt_dir, filename),
                        cv2.cvtColor(hr.numpy()[0], cv2.COLOR_RGB2BGR))
    avg_psnr = tf.reduce_mean(psnr_values)
    avg_runtime = tf.reduce_mean(runtime_values[1:]) if len(runtime_values) > 1 else tf.reduce_mean(runtime_values)
    return avg_psnr, score(avg_psnr, avg_runtime), avg_runtime


def evaluate_quantized_model(quantized_tflite_model, checkpoint_dir, valid_loader):
    psnrs = []
    assert os.path.exists(checkpoint_dir), "checkpoint_dir does not exist"
    save_result_dir = os.path.join(checkpoint_dir, f"images_result")
    save_gt_dir = os.path.join(checkpoint_dir, f"images_gt_hr")
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
    if not os.path.exists(save_gt_dir):
        os.makedirs(save_gt_dir)
    count = 800
    for data in valid_loader.dataset(batch_size=1, random_transform=True, repeat_count=1):
        if os.path.exists(quantized_tflite_model) and quantized_tflite_model.endswith('.tflite'):
            interpreter = tf.lite.Interpreter(model_path=quantized_tflite_model)
        else:
            interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        # print("interpreter tensor details", interpreter.get_tensor_details())
        gt_image = data[1].numpy().astype(interpreter.get_input_details()[0]["dtype"])[0]
        test_image = data[0].numpy().astype(interpreter.get_input_details()[0]["dtype"])
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        interpreter.set_tensor(input_index, test_image)
        # Run inference.
        interpreter.invoke()
        output = interpreter.tensor(output_index)
        output_image = output()[0]
        # print(output_image, gt_image)
        count += 1
        cv2.imwrite(os.path.join(save_result_dir, f"{str(count)}.png"), output_image)
        cv2.imwrite(os.path.join(save_gt_dir, f"{str(count)}.png"), gt_image)
        # print(count)
        # Compare prediction results with ground truth labels to calculate psnr.
        psnr_v = calculate_psnr(output_image, gt_image)
        psnrs.append(psnr_v)
    return np.mean(psnrs)

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

def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

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