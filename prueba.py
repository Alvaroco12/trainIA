from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

ruta = "mi_numero.png"
img = Image.open(ruta).convert("L")
img = ImageOps.invert(img)  # forzar inversión
img = img.resize((28,28), Image.Resampling.LANCZOS)
arr = np.array(img)/255.0
plt.imshow(arr, cmap="gray")
plt.show()
print(arr)
