import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

from cuda_image_processor import rgbtogray
from cuda_image_processor import gaussianblur
from cuda_image_processor import laplacefilter


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CUDA Image Processing")

        self.image_label = tk.Label(root)
        self.image_label.pack(padx=10, pady=10)

        frame = tk.Frame(root)
        frame.pack(pady=5)

        tk.Button(frame, text="Abrir", command=self.open_image).grid(row=0, column=0, padx=5)
        tk.Button(frame, text="Gray", command=self.apply_gray).grid(row=0, column=1, padx=5)
        tk.Button(frame, text="Blur 3x3", command=self.apply_blur).grid(row=0, column=2, padx=5)
        tk.Button(frame, text="Laplace", command=self.apply_laplace).grid(row=0, column=3, padx=5)

        self.image = None

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.png *.jpg *.jpeg")]
        )
        if not path:
            return

        img = Image.open(path).convert("RGBA")
        self.image = np.array(img)

        self.show_image(self.image)

    def show_image(self, array):
        if array.ndim == 2:
            img = Image.fromarray(array.astype(np.uint8), mode="L")
        else:
            img = Image.fromarray(array.astype(np.uint8), mode="RGBA")

        img = img.resize((400, 400))
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo)

    def apply_gray(self):
        if self.image is None:
            return

        gray = rgbtogray(self.image)
        self.image = gray
        self.show_image(gray)

    def apply_blur(self):
        if self.image is None:
            return

        blurred = gaussianblur(self.image)
        self.image = blurred
        self.show_image(blurred)

    def apply_laplace(self):
        if self.image is None:
            return

        edges = laplacefilter(self.image)
        self.image = edges
        self.show_image(edges)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
