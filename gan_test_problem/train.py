import os
import time
import numpy as np
import matplotlib.pyplot as plt
import functools
import tensorflow as tf
from tensorflow.keras import layers

from data_generation.stress_strain import *
from wgan_gp import *
from utils import *

N_SAMPLES = 100000
MAX_STRAIN = 0.02
NUM_STRAINS = 10

stress_mat, strains = generate_samples(MAX_STRAIN, NUM_STRAINS, N_SAMPLES)

EPOCHS = 150
N_SAMPLES = 1024
BATCH_SIZE = 16
ITERATIONS = int(N_SAMPLES/BATCH_SIZE)

NOISE_DIM = 3
MODEL_DIM = 512 # Model dimensionality (Dense units)
LAMBDA = .1 # Smaller lambda makes things faster for toy tasks, but isn't
            # necessary if you increase CRITIC_ITERS enough
CRITIC_ITERS = 5 # How many critic iterations per generator iteration

# Generate, standardize and shuffle data.
dataset, scaler = generate_stress_samples(MAX_STRAIN, NUM_STRAINS, N_SAMPLES, standardize)
np.random.shuffle(dataset)

# Create Generator model
generator = make_generator_model(NOISE_DIM, MODEL_DIM)
generator.summary()

# Create Discriminator model
discriminator = make_discriminator_model(NOISE_DIM, MODEL_DIM)
discriminator.summary()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)

# Discriminator Accuracy
discriminator_accuracy = tf.keras.metrics.BinaryAccuracy()

# Instantiate Generator Object
generator = Generator(generator, generator_optimizer)

# Instantiate Discriminator Object
discriminator = Discriminator(discriminator, discriminator_optimizer, discriminator_accuracy)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator.generator_optimizer, \
                                 discriminator_optimizer=discriminator.discriminator_optimizer, \
                                 generator=generator.generator, \
                                 discriminator=discriminator.discriminator)

# Training
