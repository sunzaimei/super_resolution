import datetime
import time

from dataset import DIV2K
from model.xlsr import Xlsr
from trainer import XlsrTrainer
import os
from model.common import DATASET_DIR, SCALE
import tensorflow as tf


train_loader = DIV2K(scale=SCALE, downgrade='bicubic', subset='train', mode='train')
valid_loader = DIV2K(scale=SCALE, downgrade='bicubic', subset='train', mode='valid')
# valid_loader = DIV2K(scale=SCALE, downgrade='bicubic', subset='valid')

# Create corresponding tf.data.Dataset
train_ds = train_loader.dataset(batch_size=16, random_transform=True, repeat_count=-1)
valid_ds = valid_loader.dataset(batch_size=16, random_transform=True, repeat_count=1)

xlsr = Xlsr()
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = os.path.join(DATASET_DIR, f'weights/xlsr-16-x{SCALE}_{current_time}')
trainer = XlsrTrainer(model=xlsr.xlsr(num_gblocks=3, scale=SCALE), checkpoint_dir=checkpoint_dir)

# trainer.train(train_ds, valid_ds, steps=500, evaluate_every=100, save_best_only=True)
trainer.train(train_ds, valid_ds, steps=500000, evaluate_every=100, save_best_only=True)

# Restore from checkpoint with highest PSNR.
trainer.restore()

# Evaluate model on full validation set.
psnr, score, runtime = trainer.evaluate(valid_ds)
print(f'PSNR = {psnr.numpy():3f}, SCORE = {score.numpy():3f}, RUNTIME = {runtime.numpy():3f}')

# Save model
saved_model_path = os.path.join(checkpoint_dir, 'saved_model', str(int(time.time())))
tf.saved_model.save(trainer.model, saved_model_path)

