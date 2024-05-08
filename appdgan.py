import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import filedialog
import os
import tensorflow as tf
from tensorflow.keras.layers import (Reshape, LeakyReLU, Conv2DTranspose, Conv2D, BatchNormalization, Flatten, Dense, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
import cv2
import numpy as np

BATCH_SIZE = 35
IM_SHAPE = (64, 64, 3)
LATENT_DIM = 50

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

        # Discriminator
        random_noise = tf.random.normal(shape=(batch_size, LATENT_DIM))
        fake_images = self.generator(random_noise)

        real_labels = tf.ones((batch_size, 1)) + 0.25 * tf.random.uniform((batch_size, 1), minval=-1, maxval=1)
        fake_labels = tf.zeros((batch_size, 1)) + 0.25 * tf.random.uniform((batch_size, 1), )

        with tf.GradientTape() as tape:
            real_predictions = self.discriminator(real_images)
            d_loss_real = self.loss_fn(real_labels, real_predictions)

            fake_predictions = self.discriminator(fake_images)
            d_loss_fake = self.loss_fn(fake_labels, fake_predictions)

            d_loss = d_loss_real + d_loss_fake

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Generator
        random_noise = tf.random.normal(shape=(batch_size, LATENT_DIM))
        flipped_fake_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            fake_predictions = self.discriminator(self.generator(random_noise))
            g_loss = self.loss_fn(flipped_fake_labels, fake_predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {'d_loss': self.d_loss_metric.result(),
                'g_loss': self.g_loss_metric.result()}

class GAN_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GAN Results")
        self.selected_folder = None 

        # Initialize components
        self.setup_components()

    def setup_components(self):
        # Label for folder selection
        self.folder_label = tk.Label(self.root, text="Select Folder:")
        self.folder_label.grid(column=0, row=0, padx=10, pady=10)

        # Button to select folder
        self.select_folder_button = ttk.Button(self.root, text="Select Folder", command=self.select_folder)
        self.select_folder_button.grid(column=1, row=0, padx=10, pady=10)

        # Start Training Button
        self.start_training_button = ttk.Button(self.root, text="Start Training", command=self.start_training)
        self.start_training_button.grid(column=0, row=1, columnspan=2, padx=10, pady=10)

        # Text area to display selected folder
        self.selected_folder_text = scrolledtext.ScrolledText(self.root, width=40, height=10)
        self.selected_folder_text.grid(column=0, row=2, columnspan=2, padx=10, pady=10)

    def select_folder(self):
        folder_selected = filedialog.askdirectory()
        self.selected_folder_text.delete(1.0, tk.END)  # Clear previous content
        self.selected_folder_text.insert(tk.END, f"Selected Folder: {folder_selected}")
        self.selected_folder = folder_selected

    def start_training(self):
        # Code to start training the GAN model
        print("Training started...")
        if not self.selected_folder:
            print("No folder selected. Please select a folder before starting training.")
            return

        # Load images from the selected folder
        images = []
        for file_name in os.listdir(self.selected_folder):
            file_path = os.path.join(self.selected_folder, file_name)
            if os.path.isfile(file_path):
                image = tf.io.read_file(file_path)  # Read image file
                image = tf.image.decode_image(image, channels=3)  # Decode image
                image = tf.image.resize(image, (64, 64))  # Resize image to (64, 64)
                image = tf.cast(image, tf.float32) / 255.0  # Normalize image
                images.append(image)
        
        # Create dataset from images
        train_dataset = tf.data.Dataset.from_tensor_slices(images).batch(BATCH_SIZE)

        # Initialize discriminator and generator
        discriminator = tf.keras.Sequential([
            Input(shape=(64, 64, 3)),
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
        ])

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
        ])

        # Initialize GAN model
        gan = GAN(discriminator, generator)

        # Compile GAN model
        gan.compile(
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
            loss_fn=tf.keras.losses.BinaryCrossentropy()
        )
        
        EPOCHS = 10 # Increase the number of epochs
        total_steps = 1000
        # Adjust this value as needed
        steps_per_epoch = total_steps // EPOCHS
        train_dataset = train_dataset.repeat(EPOCHS)

        history = gan.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=steps_per_epoch)


        def post_process_image(image):
            # Convert the input image to 8-bit unsigned integer format
            image_uint8 = np.clip(image, 0, 255).astype(np.uint8)

            # Apply Gaussian blur for denoising
            denoised_image = cv2.GaussianBlur(image_uint8, (5, 5), 0)
    
            # Convert the denoised image to LAB color space
            lab_image = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2LAB)

            # Split LAB image into channels
            l_channel, a_channel, b_channel = cv2.split(lab_image)

            # Apply histogram equalization to the L channel (lightness)
            equalized_l_channel = cv2.equalizeHist(l_channel)
    
            # Merge equalized L channel with original A and B channels
            equalized_lab_image = cv2.merge((equalized_l_channel, a_channel, b_channel))
    
            # Convert LAB image back to RGB color space
            enhanced_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_LAB2RGB)
    
            return enhanced_image

        # Generate images after training
        num_samples = 16
        noise = tf.random.normal(shape=(num_samples, LATENT_DIM))
        generated_images = generator.predict(noise)

        # Create a directory to save the generated images if it doesn't exist
        if not os.path.exists('generated'):
            os.makedirs('generated')
        plt.figure(figsize=(10, 10))
        for i in range(num_samples):
            plt.subplot(4, 4, i+1)
            processed_image = post_process_image((generated_images[i] + 1) * 255 / 2)  # Scale images to [0, 255] range for OpenCV
            plt.imshow(processed_image)
            plt.axis('off')
            plt.savefig("generated/gen_images_epoch_{}.png".format(EPOCHS+1)) # Save each processed image
        plt.show()

        # Display and save generated images
        '''  plt.figure(figsize=(10, 10))
        for i in range(num_samples):
            plt.subplot(4, 4, i+1)
            plt.imshow((generated_images[i] + 1) / 2,interpolation='nearest')  # Scale images back to [0, 1] range
            plt.axis('off')
            plt.savefig(f'generated/generated_image_{i}.png')  # Save each generated image
        plt.show()'''
        
        
        
# Initialize Tkinter
root = tk.Tk()
gan_gui = GAN_GUI(root)
root.mainloop()
