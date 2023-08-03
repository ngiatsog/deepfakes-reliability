import os
import numpy as np
import cv2

from inputs.sources import FaceForensics
from random import shuffle
from PIL import Image
from time import time

s = FaceForensics(small_margin=True, mode="test", qualities=["raw"])
samples = s.discriminative_samples()
shuffle(samples)
image_paths = [str(path) for path, _ in samples[:300]]
images_sizes = []
arr_paths = []
arr_sizes = []

for i, path in enumerate(image_paths):
    images_sizes.append(os.stat(path).st_size)
    with Image.open(path) as im:
        arr = np.array(im)
        arr_path = os.path.join("arrs", f"arr{i}.npz")
        arr_paths.append(arr_path)
        np.savez_compressed(arr_path, arr)
        arr_sizes.append(os.stat(arr_path).st_size)

print(f"mean size for images: {sum(images_sizes)/len(images_sizes)}")
print(f"mean size for arrays: {sum(arr_sizes)/len(arr_sizes)}")
print(f"ratio: {sum(arr_sizes) / sum(images_sizes)}")
print()

start = time()
for path in image_paths:
    # _ = cv2.imread(path)
    im = Image.open(path)
    _ = np.array(im)
    im.close()
elapsed_images = time() - start

start = time()
for arr_path in arr_paths:
    _ = np.load(arr_path)["arr_0"]
elapsed_arrays = time() - start
print(f"elapsed time for images: {elapsed_images/len(image_paths)}")
print(f"elapsed time for arrays: {elapsed_arrays/len(arr_paths)}")
print(f"ratio: {elapsed_arrays / elapsed_images}")
