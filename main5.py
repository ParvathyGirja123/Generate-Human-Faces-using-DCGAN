import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Reshape, LeakyReLU, Dropout, Conv2DTranspose, Add, Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

if tf.config.list_physical_devices('GPU'):
    # GPU is available
    device = 'GPU'
else:
    # No GPU available, fallback to CPU
    device = 'CPU'

BATCH_SIZE = 35
IM_SHAPE = (64, 64, 3)
LEARNING_RATE = 2e-4
LATENT_DIM = 50
EPOCHS = 5

dataset = tf.keras.preprocessing.image_dataset_from_directory("img/", label_mode=None, image_size=(IM_SHAPE[0], IM_SHAPE[1]), batch_size=BATCH_SIZE)

def preprocess(image):
    return tf.cast(image, tf.float32) / 127.5 - 1.0

train_dataset = (
    dataset
    .map(preprocess)
    .unbatch()
    .shuffle(buffer_size=100, reshuffle_each_iteration=True)  # Adjust the buffer size as needed
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

plt.figure(figsize=(4, 4))
for d in train_dataset.take(1):
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow((d[i] + 1) / 2)
        plt.axis('off')
plt.show()

generator = tf.keras.Sequential([
    Input(shape=(LATENT_DIM,)),
    Dense(4 * 4 * LATENT_DIM),
    Reshape((4, 4, LATENT_DIM)),
    Conv2DTranspose(512, kernel_size=4, strides=2, padding='same'),
    BatchNormalization(),
    LeakyReLU(0.2),
    Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'),
    BatchNormalization(),
    LeakyReLU(0.2),
    Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
    BatchNormalization(),
    LeakyReLU(0.2),
    Conv2DTranspose(3, kernel_size=4, strides=2, activation=tf.keras.activations.tanh, padding='same'),
], name='generator')

discriminator = tf.keras.Sequential([
    Input(shape=(IM_SHAPE[0], IM_SHAPE[1], 3)),
    Conv2D(64, kernel_size=4, strides=2, padding='same'),
    LeakyReLU(0.2),
    Conv2D(128, kernel_size=4, strides=2, padding='same'),
    BatchNormalization(),
    LeakyReLU(0.2),
    Conv2D(256, kernel_size=4, strides=2, padding='same'),
    BatchNormalization(),
    LeakyReLU(0.2),
    Conv2D(1, kernel_size=4, strides=2, padding='same'),
    Flatten(),
    Dense(1, activation='sigmoid')
], name='discriminator')

class ShowImage(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        n = 6
        k = 0
        out = self.model.generator(tf.random.normal(shape=(36, self.latent_dim)))
        plt.figure(figsize=(16, 16))
        for i in range(n):
            for j in range(n):
                ax = plt.subplot(n, n, k + 1)
                plt.imshow((out[k] + 1) / 2,)
                plt.axis('off')
                k = k + 1
        plt.savefig("generated/gen_images_epoch_{}.png".format(epoch + 1))

class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean(name='d_loss')
        self.g_loss_metric = tf.keras.metrics.Mean(name='g_loss')

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        ######## Discriminator
        random_noise = tf.random.normal(shape=(batch_size, LATENT_DIM))
        fake_images = self.generator(random_noise)

        real_labels = tf.ones((batch_size, 1)) + 0.25 * tf.random.uniform((batch_size, 1), minval=-1, maxval=1)
        fake_labels = tf.zeros((batch_size, 1)) + 0.25 * tf.random.uniform((batch_size, 1), )

        with tf.GradientTape() as recorder:
            real_predictions = self.discriminator(real_images)
            d_loss_real = self.loss_fn(real_labels, real_predictions)

            fake_predictions = self.discriminator(fake_images)
            d_loss_fake = self.loss_fn(fake_labels, fake_predictions)

            d_loss = d_loss_real + d_loss_fake

        partial_derivatives = recorder.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(partial_derivatives, self.discriminator.trainable_weights))

        ############# Generator
        random_noise = tf.random.normal(shape=(batch_size, LATENT_DIM))
        flipped_fake_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as recorder:

            fake_predictions = self.discriminator(self.generator(random_noise))
            g_loss = self.loss_fn(flipped_fake_labels, fake_predictions)

        partial_derivatives = recorder.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(partial_derivatives, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {'d_loss': self.d_loss_metric.result(),
                'g_loss': self.g_loss_metric.result()}

# Initialize optimizers for the discriminator and generator
d_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5)

# Initialize the GAN model
gan = GAN(discriminator, generator)

# Compile the GAN model with separate optimizers for discriminator and generator
gan.compile(
    d_optimizer=d_optimizer,
    g_optimizer=g_optimizer,
    loss_fn=tf.keras.losses.BinaryCrossentropy()
)

# Calculate the total number of samples in your dataset
total_samples = len(dataset)

# Calculate the total number of steps required to cover the entire dataset once
total_steps = total_samples // BATCH_SIZE

# Set the number of epochs
EPOCHS = 5

# Set the steps per epoch based on the total steps and the number of epochs
steps_per_epoch = total_steps // EPOCHS

#train_dataset = train_dataset.repeat(EPOCHS)


# Train the GAN model with the specified steps per epoch
history = gan.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[ShowImage(LATENT_DIM)])




num_samples = 16
noise = tf.random.normal(shape=(num_samples, LATENT_DIM))
generated_images = generator.predict(noise)
plt.figure(figsize=(10, 10))
for i in range(num_samples):
    plt.subplot(4, 4, i+1)
    plt.imshow((generated_images[i] + 1) / 2,interpolation='nearest')
    plt.axis('off')
plt.show()



# Plot the GAN loss history
plt.plot(history.history['d_loss'])
plt.plot(history.history['g_loss'])
plt.title('GAN Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['d_loss', 'g_loss'], loc='upper left')
plt.show()
