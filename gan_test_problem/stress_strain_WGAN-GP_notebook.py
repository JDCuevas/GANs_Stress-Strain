#%% [markdown]
# # Using a WGAN-GP to Generate Stress-strain Curves
# 
# ## Problem Definition
# 
# Assuming bi-linear stress strain behavior of a material (characterized by $\sigma_y$, $E$, and $H$), generate sample stress-strain curves based on some initial samples of a stress-strain curve distrbution. 
# 
# <img src="bilinear.png" alt="Drawing" style="width: 300px;"/>
#%% [markdown]
# ## Generating the training data
# 
# The training samples will be gathered by asuming independent, normal distributions for $\sigma_y$, $E$, and $H$.
# 
#   * $\sigma_y \sim \mathcal{N}(\mu=10, \sigma=0.5)$
#   * $E \sim \mathcal{N}(\mu=1000, \sigma=50)$
#   * $H \sim \mathcal{N}(\mu=50, \sigma=5)$

#%%
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

#%% [markdown]
# Two helper functions

#%%
def get_stress(strains, E, s_y, H):
    e_y = s_y / E
    elastic_strains = strains.copy()
    elastic_strains[elastic_strains > e_y] = e_y
    plastic_strains = strains - elastic_strains
    stresses = elastic_strains*E + plastic_strains*H
    return stresses


#%%
def generate_samples(max_strain, n_strain, n_samples):
    strain = np.linspace(0, max_strain, n_strain + 1)[1:]
    stresses = np.empty((n_samples, n_strain))
    for i in tqdm(range(n_samples), desc='Generating samples'):
        E = np.random.normal(1000, 50)
        s_y = np.random.normal(10, 0.5)
        H = np.random.normal(50, 5)
        stresses[i] = get_stress(strain, E, s_y, H)
    return stresses, strain

#%% [markdown]
# Make training data:
# 
#   * rows in stress_mat correspond to the stresses in a single stress strain curve (i.e. 1 sample)
#   * columns in stress_mat correspond to a single strain value

#%%
N_SAMPLES = 100000
MAX_STRAIN = 0.02
NUM_STRAINS = 10

stress_mat, strains = generate_samples(MAX_STRAIN, NUM_STRAINS, N_SAMPLES)

#%% [markdown]
# ## Train a GAN to produce samples that match this distribution

#%%
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import functools
from ipywidgets import interact, fixed
import ipywidgets as widgets

#%% [markdown]
# ### Hepler Functions

#%%
def standardize(data):
    scaler = StandardScaler()
    scaler.fit(data)
    standardized_data = scaler.transform(data)
    
    return standardized_data, scaler


#%%
def normalize(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    standardized_data = scaler.transform(data)
    
    return normalized_data, scaler


#%%
def generate_stress_samples(n_samples, preprocessing):
    stresses, _ = generate_samples(MAX_STRAIN, NUM_STRAINS, n_samples)
    stresses = np.array(stresses)
    
    scaled_stresses, stress_scaler = preprocessing(stresses)
    
    return scaled_stresses, stress_scaler


#%%
def plot(variable, labels, x_label, y_label, title):
    for values, label in zip(variable, labels):
        plt.plot(values, label=label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.tight_layout()


#%%
def plot_scatter(y_values, x_values, labels, x_label, y_label, title):
    for y_value, label in zip(y_values, labels):
        plt.scatter(np.expand_dims(x_values, -1), np.expand_dims(y_value, -1), label=label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.tight_layout()


#%%
def plot_hist(x_values, y_values, x_label, y_label, x):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(16, 7))

    sns.violinplot(data=pd.DataFrame(data=y_values, columns=np.round(x_values, 3)), ax=ax_left)
    ax_left.set(xlabel=x_label, ylabel=y_label)
                                                                              
    itemindex = np.argmin(abs(x_values-x))
    sns.distplot(y_values[:, itemindex], bins=20)
    ax_right.set(xlim=(np.min(y_values), np.max(y_values)), xlabel='stresses at strain of %.3f' % x_values[itemindex])    


#%%
def plot_hist_comparison(x_values, y_values, y_values_2, x_label, y_label, x):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(16, 7))

    sns.violinplot(data=pd.DataFrame(data=y_values, columns=np.round(x_values, 3)), ax=ax_left)
    ax_left.set(xlabel=x_label, ylabel=y_label)
                                                                              
    itemindex = np.argmin(abs(x_values-x))
    sns.distplot(y_values[:, itemindex], color="blue", bins=20, label='y_values')
    if np.any(y_values_2):
        sns.distplot(y_values_2[:, itemindex], color="red", bins=20, label='y_values_2')
    plt.legend()
    ax_right.set(xlim=(np.min(y_values), np.max(y_values)), xlabel='stresses at strain of %.3f' % x_values[itemindex])
    plt.savefig('data_dist.png')


#%%
def plot_all_hist(x_values, y_values, y_values_2):
    f, axes = plt.subplots(int(len(x_values)/2), int(len(x_values)/(len(x_values)/2)), figsize=(15, 15), sharex=False)
    for i in range(len(x_values)):
        sns.distplot(y_values[:, i] , color="blue", bins=20, label='y_values', ax=axes[i % int(len(x_values)/2), 0 if i < len(x_values)/2 else 1])
        sns.distplot(y_values_2[:, i] , color="red", bins=20, label='y_values_2', ax=axes[i % int(len(x_values)/2), 0 if i < len(x_values)/2 else 1])
        axes[i % int(len(x_values)/2), 0 if i < len(x_values)/2 else 1].set(xlim=(np.min(y_values), np.max(y_values)), xlabel='stresses at strain of %.3f' % x_values[i])
        
    plt.tight_layout()
    plt.savefig('data_dist.png')


#%%
def plot_individual_sample(sample, n):
    fig = plt.figure(n)
    stress_ax = plt.plot(strains, sample)
    
    plt.xlabel('strain')
    plt.ylabel('stress')
    plt.title('Stress-Strain Curve ' + str(i + 1))
    plt.tight_layout()

#%% [markdown]
# ### Visualize the training distribution

#%%
interact(plot_hist, x_values=fixed(strains), y_values=fixed(stress_mat), x_label=fixed('strain'), y_label=fixed('stress'), x=(0.0, 0.02, 0.002))

#%% [markdown]
# ### Settings

#%%
EPOCHS = 150
N_SAMPLES = 1024
BATCH_SIZE = 16
ITERATIONS = int(N_SAMPLES/BATCH_SIZE)
NOISE_DIM = 3


#%%
DIM = 512 # Model dimensionality
LAMBDA = .1 # Smaller lambda makes things faster for toy tasks, but isn't
            # necessary if you increase CRITIC_ITERS enough
CRITIC_ITERS = 5 # How many critic iterations per generator iteration

#%% [markdown]
# ### 1. Preparing the dataset

#%%
dataset, scaler = generate_stress_samples(N_SAMPLES, standardize)
np.random.shuffle(dataset)

#%% [markdown]
# ### 2. Create Models
# 
# * #### Generator

#%%
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(NOISE_DIM,)))
    model.add(layers.Dense(DIM, use_bias=False))
    #model.add(layers.LayerNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(DIM, use_bias=False))
    #model.add(layers.LayerNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(10, use_bias=False))
    
    return model
    


#%%
generator = make_generator_model()
generator.summary()

#%% [markdown]
# * #### Discriminator

#%%
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(10,)))
    model.add(layers.Dense(DIM))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(DIM))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(10))
    model.add(layers.Dense(1))
    
    return model


#%%
discriminator = make_discriminator_model()
discriminator.summary()

#%% [markdown]
# ### 3. Define loss and optimizers

#%%
accuracy = tf.keras.metrics.BinaryAccuracy()

#%% [markdown]
# * #### Gradient Penalty (TF2 Git)

#%%
def gradient_penalty(disc, real, fake):
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

#%% [markdown]
# * #### Discriminator loss

#%%
def discriminator_loss(real_output, fake_output):
    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output) 
    total_loss = real_loss + fake_loss
    return total_loss

#%% [markdown]
# * #### Discriminator accuracy

#%%
def discriminator_accuracy(real_output, fake_output):
    accuracy.update_state(tf.ones_like(real_output), real_output)
    accuracy.update_state(tf.zeros_like(fake_output), fake_output)

#%% [markdown]
# * #### Generator loss

#%%
def generator_loss(fake_output):
    loss = -tf.reduce_mean(fake_output)
    return loss

#%% [markdown]
# * #### Optimizers

#%%
generator_optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)

#%% [markdown]
# ### 4. Training
# * #### Save checkpoints

#%%
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

#%% [markdown]
# * #### Generator Training Step

#%%
@tf.function
def generator_train_step():
    with tf.GradientTape() as gen_tape:
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
        generated_batch = generator(noise, training=True)
        
        fake_output = discriminator(generated_batch, training=True)
        
        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss

#%% [markdown]
# * #### Discriminator Training Step

#%%
@tf.function
def discriminator_train_step(batch):
    with tf.GradientTape() as disc_tape:
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
        generated_batch = generator(noise, training=True)

        real_output = discriminator(batch, training=True)
        fake_output = discriminator(generated_batch, training=True)

        disc_loss = discriminator_loss(real_output, fake_output)
        gp = gradient_penalty(functools.partial(discriminator, training=True), batch, generated_batch)

        disc_loss += (gp * LAMBDA)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    discriminator_accuracy(real_output, fake_output)

    return disc_loss

#%% [markdown]
# * #### Training

#%%
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


#%%
get_ipython().run_cell_magic('time', '', 'losses_per_epoch, disc_accuracy = train(EPOCHS)')

#%% [markdown]
# ### 5. Visualize Results

#%%
num_examples = 100000
seed = tf.random.normal([num_examples, NOISE_DIM])

pred = generator(seed, training=False)
unscaled_pred = scaler.inverse_transform(pred)

samples, strains = generate_samples(MAX_STRAIN, NUM_STRAINS, num_examples)

strains_for_plot = strains * np.ones((num_examples, 1))
plot_scatter([stress_mat, unscaled_pred], strains_for_plot, ['Real Data', 'Generated Data'], x_label='strain', y_label='stress', title='Examples')

plt.savefig('Examples')

#%% [markdown]
# * #### Plotting data distribution

#%%
interact(plot_hist_comparison, x_values=fixed(strains), y_values=fixed(stress_mat), y_values_2=fixed(unscaled_pred), x_label=fixed('strain'), y_label=fixed('stress'), x=(0.0, 0.02, 0.002))


#%%
plot_all_hist(x_values=strains, y_values=stress_mat, y_values_2=unscaled_pred)

#%% [markdown]
# * #### Plotting the generator's and the discriminator's loss.

#%%
plot([losses_per_epoch[:,0], losses_per_epoch[:,1]], labels=['Discriminator Loss', 'Generator Loss'], x_label='epoch', y_label='loss', title='Network Losses')
plt.savefig('losses.png')

#%% [markdown]
# * #### Plotting Discriminator Accuracy

#%%
plot([disc_accuracy], labels=['Discriminator Accuracy'], x_label='epoch', y_label='accuracy', title=
'Discriminator Accuracy')
plt.savefig('disc_acc.png')

#%% [markdown]
# * #### Plotting individual samples

#%%
for i in range(10):
    plot_individual_sample(unscaled_pred[i], i)
    plt.savefig('example_curve.png')


