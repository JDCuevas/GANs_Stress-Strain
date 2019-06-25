import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import functools

########################################################
####################### MODELS #########################
########################################################

def make_generator_model(MODEL_DIM):
    model = tf.keras.Sequential()
    model.add(layers.Dense(MODEL_DIM, use_bias=False))
    #model.add(layers.LayerNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(MODEL_DIM, use_bias=False))
    #model.add(layers.LayerNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(10, use_bias=False))
    
    return model

class Generator:
    generator = None
    generator_optimizer = None

    def __init__(self, model=None, optimizer=None):
        self.generator = model
        self.generator_optimizer = optimizer

    @tf.function
    def generator_train_step(self, discriminator, BATCH_SIZE, NOISE_DIM):
        with tf.GradientTape() as gen_tape:
            noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
            generated_batch = self.generator(noise, training=True)
            
            fake_output = discriminator(generated_batch, training=True)
            
            gen_loss = self.generator_loss(fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return gen_loss

    def generator_loss(self, fake_output):
        loss = -tf.reduce_mean(fake_output)
        return loss

def make_discriminator_model(NOISE_DIM, MODEL_DIM):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(10,)))
    model.add(layers.Dense(MODEL_DIM))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(MODEL_DIM))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(10))
    model.add(layers.Dense(1))
    
    return model

class Discriminator:
    discriminator = None
    discriminator_optimizer = None
    discriminator_accuracy = None

    def __init__(self, model=None, optimizer=None, accuracy=None):
        self.discriminator = model
        self.discriminator_optimizer = optimizer
        self.discriminator_accuracy = accuracy

    @tf.function
    def discriminator_train_step(self, generator, batch, NOISE_DIM, LAMBDA):
        with tf.GradientTape() as disc_tape:
            noise = tf.random.normal([batch.shape[0], NOISE_DIM])
            generated_batch = generator(noise, training=True)

            real_output = self.discriminator(batch, training=True)
            fake_output = self.discriminator(generated_batch, training=True)

            disc_loss = self.discriminator_loss(real_output, fake_output)
            gp = self.gradient_penalty(functools.partial(self.discriminator, training=True), batch, generated_batch, batch.shape[0])

            disc_loss += (gp * LAMBDA)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        if self.discriminator_accuracy:
            self.discriminator_accuracy_updt(real_output, fake_output)

        return disc_loss

    def discriminator_loss(self, real_output, fake_output):
        real_loss = -tf.reduce_mean(real_output)
        fake_loss = tf.reduce_mean(fake_output) 
        total_loss = real_loss + fake_loss
        return total_loss

    def discriminator_accuracy_updt(self, real_output, fake_output):
        self.discriminator_accuracy.update_state(tf.ones_like(real_output), real_output)
        self.discriminator_accuracy.update_state(tf.zeros_like(fake_output), fake_output)

    def gradient_penalty(self, disc, real, fake, BATCH_SIZE):
        real = tf.cast(real, tf.float32)
        
        def _interpolate(a, b):
            #shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            #alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            #inter = a + alpha * (b - a)
            #inter.set_shape(a.shape)
            
            alpha = tf.random.uniform(shape=[BATCH_SIZE, 1], minval=0., maxval=1.)
            inter = alpha * a + ((1 - alpha) * b)
            
            return inter

        x = _interpolate(real, fake)
        
        with tf.GradientTape() as t:
            t.watch(x)
            pred = disc(x)
            
        grad = t.gradient(pred, x)
        #norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        #gp = tf.reduce_mean((norm - 1.)**2)

        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
        gp = tf.reduce_mean((slopes - 1) ** 2)
        
        return gp

########################################################
###################### TRAINING ########################
########################################################

def train(epochs):
    losses_per_epoch = []
    disc_accuracy = []
    
    for epoch in range(epochs):
        start = time.time()
        
        for iteration in tqdm(range(ITERATIONS)):
            iter_start = time.time()
            
            sample_batch = dataset[iteration * BATCH_SIZE : (iteration + 1) * BATCH_SIZE]
            
            disc_loss = discriminator_train_step(sample_batch)
            
            if iteration % CRITIC_ITERS == 0:
                gen_loss = generator_train_step()
                
            if (iteration + 1) % 1000 == 0:
                    print('Time for iteration {} is {:.5f} sec'.format(iteration + 1, time.time() - iter_start))
                
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            
        print('Time for epoch {} is {:.5f} sec'.format(epoch + 1, time.time() - start))
        print("Discriminator loss: {:.4f}\t Generator loss: {:.4f}\n".format(disc_loss.numpy(), gen_loss.numpy()))
        print('Discriminator accuracy: {}\n'.format(accuracy.result().numpy()))
        losses_per_epoch.append([disc_loss, gen_loss])
        disc_accuracy.append(accuracy.result().numpy())

    return np.array(losses_per_epoch), disc_accuracy