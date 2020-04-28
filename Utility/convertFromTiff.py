from PIL import Image
import numpy as np
import os
path = '/home/mbraun/NewIS/MaskedImages/2020_02_05_JZL0133_center_007_mask.tif'
rawImage = Image.open(path)
npImage = np.array(rawImage) * 255
npImage = np.array(rawImage)

visImage = Image.fromarray(np.uint8(npImage), mode='L')
visImage.save("/home/mbraun/NewIS/MaskedImages/2020_02_05_JZL0133_center_007_mask.png", 'PNG')
# os.remove(path)
print('done')