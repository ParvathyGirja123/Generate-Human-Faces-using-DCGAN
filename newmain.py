import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tensorflow as tf

# Load the pre-trained generator model
#generator = tf.keras.models.load_model('generator_model.h5')

# Function to generate images based on the seed value
def generate_images():
    seed_value = seed_entry.get()
   

# Function to display the generated image in the GUI
def display_image(image_array):
    # Convert numpy array to ImageTk format
    image = Image.fromarray((image_array * 127.5 + 127.5).astype(np.uint8))
    image = ImageTk.PhotoImage(image)

    # Display the image in the GUI
    canvas.image = image
    canvas.create_image(0, 0, anchor=tk.NW, image=image)

# Create a Tkinter GUI window
root = tk.Tk()
root.title("GAN Image Generation")
root.geometry("400x400")

# Create a frame for the seed entry and generate button
input_frame = tk.Frame(root)
input_frame.pack(pady=20)

# Label and entry for seed value
seed_label = tk.Label(input_frame, text="Seed Value:")
seed_label.grid(row=0, column=0, padx=5)

seed_entry = tk.Entry(input_frame)
seed_entry.grid(row=0, column=1, padx=5)

# Button to generate images
generate_button = tk.Button(input_frame, text="Generate Image", command=generate_images)
generate_button.grid(row=0, column=2, padx=5)

# Create a frame for displaying generated images
image_frame = tk.Frame(root)
image_frame.pack(pady=20)

# Create a canvas to display generated images
canvas = tk.Canvas(image_frame, width=64, height=64)
canvas.pack()

# Run the Tkinter event loop
root.mainloop()
