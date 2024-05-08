import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from functools import partial

# Replace this with your actual code
def train_gan():
    # Your GAN training code here
    pass

def create_gui():
    def browse_folder():
        folder_path = filedialog.askdirectory()
        if folder_path:
            entry_folder.delete(0, tk.END)
            entry_folder.insert(tk.END, folder_path)

    def start_training():
        folder_path = entry_folder.get()
        if folder_path:
            train_gan()  # Call your GAN training function
            messagebox.showinfo("Training Complete", "GAN training has completed successfully.")
        else:
            messagebox.showerror("Error", "Please select a folder containing images.")

    root = tk.Tk()
    root.title("GAN Training GUI")

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    label_folder = tk.Label(frame, text="Select Folder:")
    label_folder.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

    entry_folder = tk.Entry(frame, width=50)
    entry_folder.grid(row=0, column=1, padx=5, pady=5)

    button_browse = tk.Button(frame, text="Browse", command=browse_folder)
    button_browse.grid(row=0, column=2, padx=5, pady=5)

    button_train = tk.Button(frame, text="Start Training", command=start_training)
    button_train.grid(row=1, columnspan=3, padx=5, pady=5)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
