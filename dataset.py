import os
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from model.common import convert_3d
from config import DATASET_DIR


class DIV2K:
    def __init__(self,
                 scale=2,
                 subset='train',
                 downgrade='bicubic',
                 mode=None,
                 images_dir=os.path.join(DATASET_DIR, 'images'),
                 caches_dir=os.path.join(DATASET_DIR, 'caches')):
        """

        :param scale: one of 2, 3, 4
        :param subset: 'train' or 'valid' Training dataset are images 001 - 792, Validation dataset are images 792 - 800
        :param downgrade: one of 'bicubic', 'unknown', 'mild' or 'difficult'
        :param images_dir: sub directory with original images
        :param caches_dir: sub directory with image cache
        """

        _scales = [2, 3, 4, 8]

        if scale in _scales:
            self.scale = scale
        else:
            raise ValueError(f'scale must be in ${_scales}')

        if subset == 'train':
            self.image_ids = range(1, 801)
            if mode == 'train':
                self.image_ids = range(1, 793)
            elif mode == 'valid':
                self.image_ids = range(793, 801)
        elif subset == 'valid':
            self.image_ids = range(801, 901)
        else:
            raise ValueError("subset must be 'train' or 'valid'")

        self.mode = mode
        self.subset = subset
        self.downgrade = downgrade
        self.images_dir = images_dir
        self.caches_dir = caches_dir

    def __len__(self):
        return len(self.image_ids)

    def dataset(self, batch_size=16, repeat_count=None, random_transform=False, add_noise=False):
        """
        :param batch_size: default to 16 as indicated by xlsr paper
        :param repeat_count:  repeat iterating over images dataset, -1 or None means indefinitely
        :param random_transform:  random crop, flip, rotate, intensity scale
        :return: tf.data.Dataset
        """
        ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset(), tf.data.Dataset.from_tensor_slices(self._lr_image_files())))
        if random_transform:
            ds = ds.map(lambda lr, hr, name: random_crop(lr, hr, name, hr_crop_size=32*self.scale, scale=self.scale), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        if add_noise:
            ds = ds.map(random_guassian, num_parallel_calls=AUTOTUNE)
            # ds = ds.map(random_intensity, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def hr_dataset(self):
        ds = self._images_dataset(self._hr_image_files()).cache(self._hr_cache_file())

        if not os.path.exists(self._hr_cache_index()):
            self._populate_cache(ds, self._hr_cache_file())

        return ds

    def lr_dataset(self):
        ds = self._images_dataset(self._lr_image_files()).cache(self._lr_cache_file())

        if not os.path.exists(self._lr_cache_index()):
            self._populate_cache(ds, self._lr_cache_file())

        return ds

    def _hr_image_files(self):
        images_dir = self._hr_images_dir()
        return [os.path.join(images_dir, f'{image_id:04}.png') for image_id in self.image_ids]

    def _lr_image_files(self):
        images_dir = self._lr_images_dir()
        return [os.path.join(images_dir, f'{image_id:04}x{self.scale}.png') for image_id in self.image_ids]

    def _hr_images_dir(self):
        return os.path.join(self.images_dir, f'DIV2K_{self.subset}_HR')

    def _lr_images_dir(self):
        return os.path.join(self.images_dir, f'DIV2K_{self.subset}_LR_{self.downgrade}', f'X{self.scale}')

    def _hr_cache_file(self):
        return os.path.join(self.caches_dir, f'DIV2K_{self.subset}_{self.mode}_HR.cache')

    def _lr_cache_file(self):
        return os.path.join(self.caches_dir, f'DIV2K_{self.subset}_{self.mode}_LR_{self.downgrade}_X{self.scale}.cache')

    def _hr_cache_index(self):
        return f'{self._hr_cache_file()}.index'

    def _lr_cache_index(self):
        return f'{self._lr_cache_file()}.index'

    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
        return ds

    @staticmethod
    def _populate_cache(ds, cache_file):
        print(f'Caching decoded images in {cache_file} ...')
        for _ in ds: pass
        print(f'Cached decoded images in {cache_file}.')


# -----------------------------------------------------------
#  Transformations
# -----------------------------------------------------------

def random_crop(lr_img, hr_img, name, hr_crop_size=96, scale=2):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped, name


def random_flip(lr_img, hr_img, name):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img, name),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img), name))


def random_rotate(lr_img, hr_img, name):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn), name


def random_guassian(lr_img, hr_img, name):
    sigma = tf.random.uniform(shape=(), minval=0.01, maxval=0.1, dtype=tf.float32)
    return convert_3d(lr_img, sigma=sigma), hr_img, name


def random_intensity(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=3, dtype=tf.int32)
    from tensorflow.keras.layers.experimental.preprocessing import Rescaling
    return tf.case(
        pred_fn_pairs=[
            (tf.equal(rn, 0), lambda: (Rescaling(1)(lr_img), Rescaling(1)(hr_img))),
            (tf.equal(rn, 1), lambda: (Rescaling(0.7)(lr_img), Rescaling(0.7)(hr_img))),
            (tf.equal(rn, 2), lambda: (Rescaling(0.5)(lr_img), Rescaling(0.5)(hr_img)))],)


def resize(lr_img, hr_img):
    return tf.image.resize(lr_img, [680, 428]), tf.image.resize(hr_img, [680, 428])