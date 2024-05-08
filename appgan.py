import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Reshape, LeakyReLU, Dropout, Conv2DTranspose, Add, Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

import warnings

# Suppress TensorFlow GPU warnings

#warnings.filterwarnings("ignore", category=tf.compat.v1.logging.warning)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only allocate GPU memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

if tf.config.list_physical_devices('GPU'):
    # GPU is available
    device = 'GPU'
else:
    # No GPU available, fallback to CPU
    device = 'CPU'
BATCH_SIZE = 50
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

class GANApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GAN Training GUI")

        self.train_button = tk.Button(self.root, text="Train GAN", command=self.train_gan)
        self.train_button.pack()

        self.generated_images_label = tk.Label(self.root, text="Generated Images")
        self.generated_images_label.pack()

        self.generated_images_canvas = tk.Canvas(self.root, width=400, height=400)
        self.generated_images_canvas.pack()

        self.loss_plot_label = tk.Label(self.root, text="Loss Plot")
        self.loss_plot_label.pack()

        self.loss_plot_canvas = tk.Canvas(self.root, width=400, height=200)
        self.loss_plot_canvas.pack()

        self.figure = plt.figure(figsize=(6, 4))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.loss_plot_canvas)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Create a new window for displaying generated images
        self.generated_images_window = tk.Toplevel(self.root)
        self.generated_images_window.title("Generated Images")
        self.generated_images_canvas_window = tk.Canvas(self.generated_images_window, width=400, height=400)
        self.generated_images_canvas_window.pack()

    def train_gan(self):
        # Train the GAN model
        history = gan.fit(train_dataset, epochs=EPOCHS, callbacks=[ShowImage(LATENT_DIM)])

        # Plot loss curves
        self.plot_loss_curves(history.history['d_loss'], history.history['g_loss'])

    def update_generated_images(self, images):
        self.generated_images_canvas.delete("all")
        self.generated_images_canvas_window.delete("all")  # Clear previous images in the window
        for i in range(len(images)):
            # Display generated images in both canvases
            image = (images[i] + 1) / 2  # Assuming images are normalized between -1 and 1
            image = np.uint8(image * 255)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image)
            self.generated_images_canvas.create_image((i % 4) * 100 + 50, (i // 4) * 100 + 50, image=photo)
            self.generated_images_canvas.image = photo
            
            # Display generated images in the window
            photo_window = ImageTk.PhotoImage(image)
            self.generated_images_canvas_window.create_image((i % 4) * 100 + 50, (i // 4) * 100 + 50, image=photo_window)
            self.generated_images_canvas_window.image = photo_window

    def plot_loss_curves(self, d_loss, g_loss):
        self.ax.clear()
        epochs = range(1, len(d_loss) + 1)
        self.ax.plot(epochs, d_loss, 'b', label='Discriminator Loss')
        self.ax.plot(epochs, g_loss, 'r', label='Generator Loss')
        self.ax.set_title('GAN Loss')
        self.ax.set_xlabel('Epochs')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = GANApp(root)
    root.mainloop()

