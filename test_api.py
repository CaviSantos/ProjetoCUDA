import numpy as np
from cuda_image_processor import rgbtogray
from tkinter import filedialog
from PIL import Image, ImageTk

# img = np.zeros((100, 100, 3), dtype=np.uint8)

try:
    path = filedialog.askopenfilename(
        filetypes=[("Imagens", "*.png *.jpg *.jpeg")]
    )

    img = Image.open(path).convert("RGB")
    
    np_array = np.array(img)
    
    print(np_array)
    print("Cavi fechado com Bolsonaro")
    print(rgbtogray(np_array))
except NotImplementedError as e:
    print("API OK:", e)
