def discriminator_loss(real_output, fake_output):
    real_loss = -tf.reduce_mean(real_output)
	fake_loss = tf.reduce_mean(fake_output) 
	total_loss = real_loss + fake_loss

	return total_loss

def discriminator_accuracy(real_output, fake_output, metric):
    metric.update_state(tf.ones_like(real_output), real_output)
	metric.update_state(tf.zeros_like(fake_output), fake_output)

def generator_train_step():
    with tf.GradientTape() as gen_tape:
	        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
			generated_batch = generator(noise, training=True)
					        
			fake_output = discriminator(generated_batch, training=True)

			gen_loss = generator_loss(fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
														    
	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

	return gen_loss

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
