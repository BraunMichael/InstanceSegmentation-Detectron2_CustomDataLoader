import os
import imageio
import random
import numpy as np  # (pip install numpy)
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from tkinter import Tk, filedialog
from console_progressbar import ProgressBar  # pip install console-progressbar
import multiprocessing
import joblib
from joblib import Parallel, delayed
import contextlib
from tqdm import tqdm
from imgaug.augmentables.batches import UnnormalizedBatch
from collections import deque
from ttictoc import tic, toc
from Utility.Utilities import *


# Parallelization implemented on ShotNoise, everything else was slower with the number of images used so far.
# Formatting of ShotNoise can be copied easily into the others if wanting to switch back to parallelization
PARALLEL_PROCESSING = True
showPlots = False  # Will show tiled grid if true, only works with a small number of images
saveFiles = True

additiveAugmentMode = True  # False is "multiplicative" mode, where there will be an image that has every augment applied, and every possible combination of augment is represented
trainingPercent = 90
validationPercent = 100 - trainingPercent
showGridImage = False


# noinspection PyShadowingNames
def saveImageAndMask(image, maskImage, augmentedRawImagesDir, augmentedMaskedImagesDir, baseFileName, baseFileType, maskFileType, fileNumber):
    imageio.imwrite(os.path.join(augmentedRawImagesDir, baseFileName + '_' + str(fileNumber) + '.' + baseFileType),
                    image)
    imageio.imwrite(
        os.path.join(augmentedMaskedImagesDir, baseFileName + '_mask_' + str(fileNumber) + '.' + maskFileType),
        maskImage.draw(colors=colorList)[0])


def getFilesInFolderList(titleString, fileTypeString):
    filesFolder = getFileOrDir('folder', titleString)
    (dirpath, dirnames, rawFileNames) = next(os.walk(filesFolder))
    fileNames = []

    for name in rawFileNames:
        if name.endswith(fileTypeString):
            fileNames.append(os.path.join(dirpath, name))
    return fileNames


def convertGrayscaleTo3ChannelFormat(filePath):
    inputImage = imageio.imread(filePath)
    if len(inputImage.shape) == 3 and inputImage.shape[2] == 3:  # The input image may already be the right shape
        return inputImage
    outputImage = np.stack((inputImage,) * 3, axis=-1)
    return outputImage


def expandList(baseImageListFunc, baseMaskListFunc, x00percent):
    alteredImageListFunc = baseImageListFunc * x00percent
    alteredMaskListFunc = baseMaskListFunc * x00percent
    return alteredImageListFunc, alteredMaskListFunc


# Define imgaug augmentations
def dictHorizontalFlip(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc):
    print('Horizontal Flip, starting number of images:', len(segmapListFunc))
    horizontalFlip_x00percent = 1
    horizontalFlip = iaa.HorizontalFlip(1)
    alteredImageListFunc, alteredMaskListFunc = expandList(baseImageListFunc, baseMaskListFunc,
                                                           horizontalFlip_x00percent)
    (alteredImageListFunc, alteredMaskListFunc) = horizontalFlip(images=alteredImageListFunc,
                                                                 segmentation_maps=alteredMaskListFunc)

    fullImageListFunc.extend(alteredImageListFunc)
    segmapListFunc.extend(alteredMaskListFunc)
    return fullImageListFunc, segmapListFunc


def dictCropMultiples(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc):
    cropMultiples_heightmultiple = 2
    cropMultiples_widthmultiple = 2
    cropMultiples_x00percent = cropMultiples_heightmultiple * cropMultiples_widthmultiple
    cropMultiples = iaa.CropToMultiplesOf(height_multiple=cropMultiples_heightmultiple,
                                          width_multiple=cropMultiples_widthmultiple)
    if PARALLEL_PROCESSING:
        batches = [UnnormalizedBatch(images=baseImageListFunc, segmentation_maps=baseMaskListFunc) for _ in
                   range(cropMultiples_x00percent)]
        batches_aug = list(cropMultiples.augment_batches(batches, background=True))
        for entry in batches_aug:
            fullImageListFunc.extend(entry.images_aug)
            segmapListFunc.extend(entry.segmentation_maps_aug)
    else:
        alteredImageListFunc, alteredMaskListFunc = expandList(baseImageListFunc, baseMaskListFunc,
                                                               cropMultiples_x00percent)
        (alteredImageListFunc, alteredMaskListFunc) = cropMultiples(images=alteredImageListFunc,
                                                                    segmentation_maps=alteredMaskListFunc)

        fullImageListFunc.extend(alteredImageListFunc)
        segmapListFunc.extend(alteredMaskListFunc)
    return fullImageListFunc, segmapListFunc


def dictShotNoise(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc):
    print('Shot noise, starting number of images:', len(segmapListFunc))
    shotNoise_x00percent = 5
    shotNoise = iaa.imgcorruptlike.ShotNoise(severity=1)

    if PARALLEL_PROCESSING:
        batches = [UnnormalizedBatch(images=baseImageListFunc, segmentation_maps=baseMaskListFunc) for _ in
                   range(shotNoise_x00percent)]
        batches_aug = list(shotNoise.augment_batches(batches, background=True))
        for entry in batches_aug:
            fullImageListFunc.extend(entry.images_aug)
            segmapListFunc.extend(entry.segmentation_maps_aug)
    else:
        alteredImageListFunc, alteredMaskListFunc = expandList(baseImageListFunc, baseMaskListFunc,
                                                               shotNoise_x00percent)
        (alteredImageListFunc, alteredMaskListFunc) = shotNoise(images=alteredImageListFunc,
                                                                segmentation_maps=alteredMaskListFunc)

        fullImageListFunc.extend(alteredImageListFunc)
        segmapListFunc.extend(alteredMaskListFunc)

    return fullImageListFunc, segmapListFunc


def dictGaussianBlur(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc):
    print('Gaussian blur, starting number of images:', len(segmapListFunc))
    gaussianBlur_x00percent = 1
    gaussianBlur = iaa.GaussianBlur(sigma=1)
    alteredImageListFunc, alteredMaskListFunc = expandList(baseImageListFunc, baseMaskListFunc, gaussianBlur_x00percent)
    (alteredImageListFunc, alteredMaskListFunc) = gaussianBlur(images=alteredImageListFunc,
                                                               segmentation_maps=alteredMaskListFunc)

    fullImageListFunc.extend(alteredImageListFunc)
    segmapListFunc.extend(alteredMaskListFunc)
    return fullImageListFunc, segmapListFunc


def dictMultiplyAug(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc):
    print('Brightness multiply, starting number of images:', len(segmapListFunc))

    multiplyAug_x00percent = 4
    multiplyAug = iaa.Multiply((0.3, 1.8))
    alteredImageListFunc, alteredMaskListFunc = expandList(baseImageListFunc, baseMaskListFunc, multiplyAug_x00percent)
    (alteredImageListFunc, alteredMaskListFunc) = multiplyAug(images=alteredImageListFunc,
                                                              segmentation_maps=alteredMaskListFunc)

    fullImageListFunc.extend(alteredImageListFunc)
    segmapListFunc.extend(alteredMaskListFunc)
    return fullImageListFunc, segmapListFunc


def dictLinearContrast(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc):
    print('Contrast, starting number of images:', len(segmapListFunc))
    linearContrast_x00percent = 4
    linearContrast = iaa.LinearContrast((0.4, 1.6))
    alteredImageListFunc, alteredMaskListFunc = expandList(baseImageListFunc, baseMaskListFunc,
                                                           linearContrast_x00percent)
    (alteredImageListFunc, alteredMaskListFunc) = linearContrast(images=alteredImageListFunc,
                                                                 segmentation_maps=alteredMaskListFunc)

    fullImageListFunc.extend(alteredImageListFunc)
    segmapListFunc.extend(alteredMaskListFunc)
    return fullImageListFunc, segmapListFunc


def dictRotateAug(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc):
    print('Rotation, starting number of images:', len(segmapListFunc))
    rotateAug_x00percent = 2
    rotateAug = iaa.Rotate((-3, 3), mode="reflect")
    rotateAug._mode_segmentation_maps = "reflect"
    alteredImageListFunc, alteredMaskListFunc = expandList(baseImageListFunc, baseMaskListFunc, rotateAug_x00percent)
    (alteredImageListFunc, alteredMaskListFunc) = rotateAug(images=alteredImageListFunc,
                                                            segmentation_maps=alteredMaskListFunc)

    fullImageListFunc.extend(alteredImageListFunc)
    segmapListFunc.extend(alteredMaskListFunc)
    return fullImageListFunc, segmapListFunc


def dictScalingAug(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc):
    print('Scaling, starting number of images:', len(segmapListFunc))
    scalingAug_x00percent = 2
    scalingAug = iaa.Affine(scale=(0.5, 1.5), mode="reflect")
    scalingAug._mode_segmentation_maps = "reflect"
    alteredImageListFunc, alteredMaskListFunc = expandList(baseImageListFunc, baseMaskListFunc, scalingAug_x00percent)
    (alteredImageListFunc, alteredMaskListFunc) = scalingAug(images=alteredImageListFunc,
                                                             segmentation_maps=alteredMaskListFunc)

    fullImageListFunc.extend(alteredImageListFunc)
    segmapListFunc.extend(alteredMaskListFunc)
    return fullImageListFunc, segmapListFunc


def dictTranslateX(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc):
    print('TranslateX, starting number of images:', len(segmapListFunc))
    translateX_x00percent = 2
    translateX = iaa.TranslateY(percent=(-0.2, 0.2), mode="reflect")
    translateX._mode_segmentation_maps = "reflect"
    alteredImageListFunc, alteredMaskListFunc = expandList(baseImageListFunc, baseMaskListFunc, translateX_x00percent)
    (alteredImageListFunc, alteredMaskListFunc) = translateX(images=alteredImageListFunc,
                                                             segmentation_maps=alteredMaskListFunc)

    fullImageListFunc.extend(alteredImageListFunc)
    segmapListFunc.extend(alteredMaskListFunc)
    return fullImageListFunc, segmapListFunc


def dictTranslateY(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc):
    print('TranslateY, starting number of images:', len(segmapListFunc))
    translateY_x00percent = 2
    translateY = iaa.TranslateX(percent=(-0.2, 0.2), mode="reflect")
    translateY._mode_segmentation_maps = "reflect"
    alteredImageListFunc, alteredMaskListFunc = expandList(baseImageListFunc, baseMaskListFunc, translateY_x00percent)
    (alteredImageListFunc, alteredMaskListFunc) = translateY(images=alteredImageListFunc,
                                                             segmentation_maps=alteredMaskListFunc)

    fullImageListFunc.extend(alteredImageListFunc)
    segmapListFunc.extend(alteredMaskListFunc)
    return fullImageListFunc, segmapListFunc


def dictShearX(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc):
    print('ShearX, starting number of images:', len(segmapListFunc))
    shearX_x00percent = 2
    shearX = iaa.ShearX((-3, 3), mode="reflect")
    shearX._mode_segmentation_maps = "reflect"
    alteredImageListFunc, alteredMaskListFunc = expandList(baseImageListFunc, baseMaskListFunc, shearX_x00percent)
    (alteredImageListFunc, alteredMaskListFunc) = shearX(images=alteredImageListFunc,
                                                         segmentation_maps=alteredMaskListFunc)

    fullImageListFunc.extend(alteredImageListFunc)
    segmapListFunc.extend(alteredMaskListFunc)
    return fullImageListFunc, segmapListFunc


def dictShearY(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc):
    print('ShearY, starting number of images:', len(segmapListFunc))
    shearY_x00percent = 2
    shearY = iaa.ShearY((-3, 3), mode="reflect")
    shearY._mode_segmentation_maps = "reflect"
    alteredImageListFunc, alteredMaskListFunc = expandList(baseImageListFunc, baseMaskListFunc, shearY_x00percent)
    (alteredImageListFunc, alteredMaskListFunc) = shearY(images=alteredImageListFunc,
                                                         segmentation_maps=alteredMaskListFunc)

    fullImageListFunc.extend(alteredImageListFunc)
    segmapListFunc.extend(alteredMaskListFunc)
    return fullImageListFunc, segmapListFunc


def imageAugment(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc, augmentationStr):
    if augmentationStr.lower() == "HorizontalFlip".lower():
        return dictHorizontalFlip(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc)
    elif augmentationStr.lower() == "CropMultiples".lower():
        return dictCropMultiples(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc)
    elif augmentationStr.lower() == "ShotNoise".lower():
        return dictShotNoise(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc)
    elif augmentationStr.lower() == "GaussianBlur".lower():
        return dictGaussianBlur(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc)
    elif augmentationStr.lower() == "MultiplyAug".lower():
        return dictMultiplyAug(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc)
    elif augmentationStr.lower() == "LinearContrast".lower():
        return dictLinearContrast(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc)
    elif augmentationStr.lower() == "RotateAug".lower():
        return dictRotateAug(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc)
    elif augmentationStr.lower() == "ScalingAug".lower():
        return dictScalingAug(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc)
    elif augmentationStr.lower() == "TranslateX".lower():
        return dictTranslateX(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc)
    elif augmentationStr.lower() == "TranslateY".lower():
        return dictTranslateY(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc)
    elif augmentationStr.lower() == "shearX".lower():
        return dictShearX(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc)
    elif augmentationStr.lower() == "shearY".lower():
        return dictShearY(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc)
    else:
        print("The augmentationStr was not found in the imageAugment def. Ending program")
        quit()


binaryMaskFileNames = sorted(getFilesInFolderList("Select Binary Mask Image Folder", ".png"))
rawImageFileNames = sorted(getFilesInFolderList("Select Raw Image Folder", ".jpg"))
num_cores = multiprocessing.cpu_count()

if len(binaryMaskFileNames) != len(rawImageFileNames):
    print(
        "There are a different number of files in each of the selected folders. Make sure you have a masked image for each raw image!")
    quit()

# Still don't think we should need 256 colors, but this appears to work well
# Actually, this might be because the mask areas are indicated by a value of 255 not 1, so it is assigning it to a class number 255
colorList = [(i, i, i) for i in range(256)]
colorList.insert(0, (0, 0, 0))

annotations = []
annotation_id = 1
image_id = 1
for entry in range(0, len(rawImageFileNames)):
    imageFileName = os.path.basename(rawImageFileNames[entry])
    baseFileType = str(imageFileName).split('.')[-1]
    baseFileName = str(imageFileName).split('.')[0]
    maskFileName = os.path.basename(binaryMaskFileNames[entry])
    maskFileType = str(maskFileName).split('.')[-1]
    print("Processing image number", entry+1)
    binaryMask = convertGrayscaleTo3ChannelFormat(binaryMaskFileNames[entry])
    originalImage = convertGrayscaleTo3ChannelFormat(rawImageFileNames[entry])
    segmap = SegmentationMapsOnImage(binaryMask, shape=originalImage.shape)
    baseImageList = [originalImage]
    baseMaskList = [segmap]

    baseImageList, baseMaskList = imageAugment(baseImageList, baseMaskList, baseImageList, baseMaskList, "HorizontalFlip")

    if additiveAugmentMode:
        fullImageList = baseImageList.copy()
        segmapList = baseMaskList.copy()
    else:
        fullImageList = baseImageList
        segmapList = baseMaskList

    # These augmentations can be in any order, and can be commented out if not desired
    fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "MultiplyAug")
    fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "RotateAug")
    fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "LinearContrast")
    # fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "TranslateX")
    # fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "TranslateY")
    fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "GaussianBlur")
    # fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "ScalingAug")
    fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "ShotNoise")
    # fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "ShearX")
    # fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "ShearY")

    if showPlots or showGridImage:
        def drawSegmentationMapsOnImages(fullImage, segmapImage, colorList):
            combinedImage = segmapImage.draw_on_image(fullImage, colors=colorList)[0]
            plt.subplots(figsize=(10, 8))
            plt.imshow(combinedImage, 'jet', interpolation='none')
            plt.show()
            return combinedImage
        if PARALLEL_PROCESSING:
            with tqdm_joblib(tqdm(desc="Drawing Segmentation Maps on Images", total=len(fullImageList))) as progress_bar:
                plotList = joblib.Parallel(n_jobs=num_cores)(
                    joblib.delayed(drawSegmentationMapsOnImages)(fullImage, segmapImage, colorList) for fullImage, segmapImage
                    in zip(fullImageList, segmapList))
        else:
            plotList = []
            for fullImage, segmapImage in zip(fullImageList, segmapList):
                plotList.append(drawSegmentationMapsOnImages(fullImage, segmapImage, colorList))

    if saveFiles:
        augmentedRawImagesDir = 'NoSnTopDownAugmentedRawImages'
        if not os.path.isdir(augmentedRawImagesDir):
            os.mkdir(augmentedRawImagesDir)
        augmentedRawImagesTrainDir = os.path.join(augmentedRawImagesDir, 'Train')
        if not os.path.isdir(augmentedRawImagesTrainDir):
            os.mkdir(augmentedRawImagesTrainDir)
        augmentedRawImagesValidationDir = os.path.join(augmentedRawImagesDir, 'Validation')
        if not os.path.isdir(augmentedRawImagesValidationDir):
            os.mkdir(augmentedRawImagesValidationDir)

        augmentedMaskedImagesDir = 'NoSnTopDownAugmentedMaskImages'
        if not os.path.isdir(augmentedMaskedImagesDir):
            os.mkdir(augmentedMaskedImagesDir)
        augmentedMaskedImagesTrainDir = os.path.join(augmentedMaskedImagesDir, 'Train')
        if not os.path.isdir(augmentedMaskedImagesTrainDir):
            os.mkdir(augmentedMaskedImagesTrainDir)
        augmentedMaskedImagesValidationDir = os.path.join(augmentedMaskedImagesDir, 'Validation')
        if not os.path.isdir(augmentedMaskedImagesValidationDir):
            os.mkdir(augmentedMaskedImagesValidationDir)

        fullImageListLength = len(fullImageList)
        trainImageListLength = round(fullImageListLength * (trainingPercent / 100))
        validationImageListLength = fullImageListLength - trainImageListLength

        fullImageListnp = np.asarray(fullImageList)
        segmapListnp = np.asarray(segmapList)

        np.random.seed(0)
        np.random.shuffle(fullImageListnp)
        fullImageListdeque = deque(fullImageListnp)

        np.random.seed(0)
        np.random.shuffle(segmapListnp)
        segmapListdeque = deque(segmapListnp)

        trainImageList = list(deque(fullImageListdeque.popleft() for _ in range(trainImageListLength)))
        validationImageList = list(fullImageListdeque)

        trainSegmapList = list(deque(segmapListdeque.popleft() for _ in range(trainImageListLength)))
        validationSegmapList = list(segmapListdeque)

        if PARALLEL_PROCESSING:
            with tqdm_joblib(tqdm(desc="Saving Training Images", total=trainImageListLength)) as progress_bar:
                Parallel(n_jobs=num_cores)(
                    delayed(saveImageAndMask)(image, maskImage, augmentedRawImagesTrainDir, augmentedMaskedImagesTrainDir, baseFileName,
                                              baseFileType, maskFileType, fileNum) for (image, maskImage, fileNum) in zip(trainImageList, trainSegmapList, list(range(1, trainImageListLength + 1))))
            with tqdm_joblib(tqdm(desc="Saving Validation Images", total=validationImageListLength)) as progress_bar:
                Parallel(n_jobs=num_cores)(
                    delayed(saveImageAndMask)(image, maskImage, augmentedRawImagesValidationDir, augmentedMaskedImagesValidationDir, baseFileName,
                                              baseFileType, maskFileType, fileNum) for (image, maskImage, fileNum) in zip(validationImageList, validationSegmapList, list(range(1, validationImageListLength + 1))))
        else:
            print("Saving Training Images")
            progressBar = ProgressBar(total=trainImageListLength)
            for (image, maskImage, fileNum) in zip(trainImageList, trainSegmapList, list(range(1, trainImageListLength + 1))):
                progressBar.print_progress_bar(fileNum)
                saveImageAndMask(image, maskImage, augmentedRawImagesTrainDir, augmentedMaskedImagesTrainDir, baseFileName, baseFileType, maskFileType, fileNum)
            print("Saving Validation Images")
            progressBar = ProgressBar(total=validationImageListLength)
            for (image, maskImage, fileNum) in zip(validationImageList, validationSegmapList, list(range(1, validationImageListLength + 1))):
                progressBar.print_progress_bar(fileNum)
                saveImageAndMask(image, maskImage, augmentedRawImagesValidationDir, augmentedMaskedImagesValidationDir, baseFileName, baseFileType, maskFileType, fileNum)

    if showGridImage:
        grid_image = ia.draw_grid(plotList)
        ia.imshow(grid_image)
        if saveFiles:
            imageio.imwrite("Grid.jpg", grid_image)