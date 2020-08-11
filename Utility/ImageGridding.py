import os
from Utility.Utilities import *
from Utility.CropScaleSave import getNakedNameFromFilePath
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


def splitSingleImage(imageName, dirpath, gridSize: int):
    rawImage = Image.open(imageName)
    (width, height) = rawImage.size
    if not width % gridSize == 0:
        rawImage = rawImage.crop((0, 0, width - (width % gridSize), height))
    if not height % gridSize == 0:
        rawImage = rawImage.crop((0, 0, width, height - (height % gridSize)))
    (width, height) = rawImage.size
    nakedFileName, fileExtension = getNakedNameFromFilePath(imageName, True)
    for rowNum in range(gridSize):
        for colNum in range(gridSize):
            croppedImage = rawImage.crop((colNum * width / gridSize, rowNum * height / gridSize, ((colNum + 1) * width / gridSize) - 1, ((rowNum + 1) * height / gridSize) - 1))
            croppedImage.save(os.path.join(dirpath, nakedFileName + "_" + str(colNum) + str(rowNum) + fileExtension))
    rawImage.close()
    os.remove(os.path.join(dirpath, nakedFileName + fileExtension))


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
        joblib.Parallel(n_jobs=num_cores)(joblib.delayed(splitSingleImage)(imageName, dirpath, gridSize) for imageName in rawImageNames)


if __name__ == "__main__":
    main()