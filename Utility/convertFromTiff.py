from PIL import Image
import numpy as np
import os
path = '/home/mbraun/InstanceSegmentation-Detectron2/MaskedImages/2020_02_24_MB0234_Andrew_002_cropped_brightened_mask.tif'
rawImage = Image.open(path)
npImage = np.array(rawImage) * 255
npImage = np.array(rawImage)

visImage = Image.fromarray(np.uint8(npImage), mode='L')
visImage.save("/home/mbraun/InstanceSegmentation-Detectron2/MaskedImages/2020_02_24_MB0234_Andrew_002_cropped_brightened_mask.png", 'PNG')
# os.remove(path)
print('done')