import os
from Utility.Utilities import *
from Utility.CropScaleSave import getNakedNameFromFilePath, getDataBarPixelRow
from PIL import Image
from tqdm import tqdm
import multiprocessing

gridSize = 4
num_cores = multiprocessing.cpu_count()


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback:
        def __init__(self, time, index, parallel):
            self.index = index
            self.parallel = parallel

        def __call__(self, index):
            tqdm_object.update()
            if self.parallel._original_iterator is not None:
                self.parallel.dispatch_next()

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def splitSingleImage(image, imagePath, gridSize: int, saveSplitImages: bool = True, deleteOriginalImage: bool = False, removeDataBar: bool = False):
    (width, height) = image.size
    if not width % gridSize == 0:
        image = image.crop((0, 0, width - (width % gridSize), height))
    if not height % gridSize == 0:
        image = image.crop((0, 0, width, height - (height % gridSize)))
    (width, height) = image.size

    fileTypeEnding = imagePath[imagePath.rfind('.'):]
    pngPath = imagePath.replace(fileTypeEnding, '.png')

    griddedImageList = []
    for rowNum in range(gridSize):
        for colNum in range(gridSize):
            gridImage = image.crop((colNum * width / gridSize, rowNum * height / gridSize, ((colNum + 1) * width / gridSize) - 1, ((rowNum + 1) * height / gridSize) - 1))
            if saveSplitImages:
                fileTypeEnding = pngPath[imagePath.rfind('.'):]
                gridImagePath = pngPath.replace(fileTypeEnding, '_' + str(colNum) + str(rowNum) + '.png')
                gridImage.save(gridImagePath)
            griddedImageList.append(gridImage)
    image.close()
    if deleteOriginalImage:
        os.remove(imagePath)
    return griddedImageList


def create_collage(listofimages, marginFraction):
    assert np.sqrt(len(listofimages)).is_integer(), "You have a non-square array of images (length of image list is not a perfect square)"
    gridSize = int(np.sqrt(len(listofimages)))
    firstImage = Image.fromarray(np.uint8(listofimages[0]), mode='RGB')
    size = firstImage.size
    (thumbnail_width, thumbnail_height) = size
    marginWidth = int(marginFraction*thumbnail_width)
    marginHeight = int(marginFraction*thumbnail_height)
    width = gridSize*thumbnail_width + (gridSize - 1)*marginWidth
    height = gridSize*thumbnail_height + (gridSize - 1)*marginHeight
    collageImage = Image.new('RGB', (width, height), 'white')
    ims = []
    for p in listofimages:
        im = Image.fromarray(np.uint8(p), mode='RGB')
        im.thumbnail(size)
        ims.append(im)
    i = 0
    x = 0
    y = 0
    for row in range(gridSize):
        for col in range(gridSize):
            # print(i, x, y)
            collageImage.paste(ims[i], (x, y))
            i += 1
            x += thumbnail_width + marginWidth
        y += thumbnail_height + marginHeight
        x = 0
    return collageImage


def main():
    filesFolder = getFileOrDir('folder', 'Choose folder of image files')
    if not filesFolder:
        print("No folder")
        quit()

    (dirpath, _, rawFileNames) = next(os.walk(filesFolder))
    rawImageNames = []
    for name in rawFileNames:
        if name.endswith(('.png', '.jpg', '.jpeg')):
            rawImageNames.append(os.path.join(dirpath, name))

    totalImages = len(rawImageNames)
    with tqdm_joblib(tqdm(desc="Splitting Images", total=totalImages)) as progress_bar:
        joblib.Parallel(n_jobs=num_cores)(joblib.delayed(splitSingleImage)(imageName, dirpath, gridSize, saveSplitImages=True, deleteOriginalImage=True) for imageName in rawImageNames)


if __name__ == "__main__":
    main()