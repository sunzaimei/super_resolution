import datetime
import math
import os
import time
import tensorflow as tf

from model.common import evaluate
# from model import srgan

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import tensorflow.python.ops.math_ops as math_ops
from model.common import DATASET_DIR

class TriCyclicLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = -1

    def __call__(self, step):
        epoch = tf.math.floor(step/100)
        self.learning_rate = tf.cond(tf.greater(epoch, 50),
                                     lambda: 25e-4 - 4.85e-7 * (epoch-50),
                                     lambda: self.initial_learning_rate + 5e-5 * epoch)
        # tf.print("step", step, "epoch", epoch, "learning_rate", self.learning_rate)
        return self.learning_rate


class CharbonnierLoss(tf.keras.losses.Loss):
    def __init__(self, eps=0.1):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(math_ops.sqrt(math_ops.square(y_pred - y_true)+math_ops.square(self.eps)), axis=-1)


class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_dir='./ckpt/edsr'):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,
                                                             epsilon=1e-08), model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        self.restore()
        train_log_dir = os.path.join(checkpoint_dir, 'gradient_tape/train')
        eval_log_dir = os.path.join(checkpoint_dir, 'gradient_tape/eval')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=100, save_best_only=False):
        """
        :param train_dataset:
        :param valid_dataset:
        :param steps: total 5000 epochs with 100 iterations each
        :param evaluate_every: save checkpoint every epoch, which is 100 iterations
        :param save_best_only: Save a checkpoint only if evaluation PSNR has improved, thus remaining to be the best known
        :return:
        """
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()
        tf.print(steps, ckpt.step.numpy())
        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            # tf.print(lr.shape, hr.shape)
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()
            loss = self.train_step(lr, hr)
            loss_mean(loss)

            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_mean.result(), step=step)

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                # Compute PSNR on validation dataset
                psnr_value, score_value, runtime_value = self.evaluate(valid_dataset)
                # psnr_value_train, score_value_train, runtime_value_train = self.evaluate(train_dataset.take(100))
                duration = time.perf_counter() - self.now
                print(
                    f'Epoch {int(step/100)}/{int(steps/100)} - {step}/{steps}: loss = {loss_value.numpy():.3f}, '
                    f'psnr = {psnr_value.numpy():3f} ({duration:.2f}s), '
                    # f'score = {score_value.numpy():3f}, '
                    f'runtime_value = {runtime_value:3f}')

                # with self.train_summary_writer.as_default():
                #     tf.summary.scalar('psnr', psnr_value_train, step=step/100)

                with self.eval_summary_writer.as_default():
                    # tf.summary.scalar('learning_rate', self.checkpoint.optimizer.lr.learning_rate, step=step / 100)
                    tf.summary.scalar('psnr', psnr_value, step=step/100)
                    # tf.summary.scalar('score', score_value, step=step)
                    tf.summary.scalar('runtime', runtime_value, step=step/100)

                if save_best_only and psnr_value <= ckpt.psnr:
                    self.now = time.perf_counter()
                    # skip saving checkpoint, no PSNR improvement
                    continue

                ckpt.psnr = psnr_value
                ckpt_mgr.save()

                self.now = time.perf_counter()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value

    def evaluate(self, dataset, save_result=False, checkpoint_dir=''):
        return evaluate(self.checkpoint.model, dataset, save_result=save_result, checkpoint_dir=checkpoint_dir)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint).expect_partial()
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')

class EdsrTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class XlsrTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=TriCyclicLRSchedule(5e-5)):
                 # learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=CharbonnierLoss(0.1), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)
