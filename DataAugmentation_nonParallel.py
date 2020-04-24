import os
import imageio
import numpy as np  # (pip install numpy)
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from tkinter import Tk, filedialog
from console_progressbar import ProgressBar  # pip install console-progressbar

# Now need to combine with other prep file and include bounding boxes https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html
showPlots = False  # Will show tiled grid if true, only works with a small number of images
saveFiles = True
additiveAugmentMode = True  # False is "multiplicative" mode, where there will be an image that has every augment applied, and every possible combination of augment is represented

# This shouldn't be needed, but otherwise an error pops up in imgaug (used to make a colormap)
numberRegions = 500


def getFilesInFolderList(titleString, fileTypeString):
    filesFolder = filedialog.askdirectory(initialdir="/home/mbraun/NewIS", title=titleString)
    if not filesFolder:
        quit()
    (dirpath, dirnames, rawFileNames) = next(os.walk(filesFolder))
    fileNames = []

    for name in rawFileNames:
        if name.endswith(fileTypeString):
            fileNames.append(os.path.join(dirpath, name))
    return fileNames


def convertGrayscaleTo3ChannelFormat(filePath):
    inputImage = imageio.imread(filePath)
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
    alteredImageListFunc, alteredMaskListFunc = expandList(baseImageListFunc, baseMaskListFunc,
                                                           cropMultiples_x00percent)
    (alteredImageListFunc, alteredMaskListFunc) = cropMultiples(images=alteredImageListFunc,
                                                                segmentation_maps=alteredMaskListFunc)

    fullImageListFunc.extend(alteredImageListFunc)
    segmapListFunc.extend(alteredMaskListFunc)
    return fullImageListFunc, segmapListFunc


def dictShotNoise(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc):
    print('Shot noise, starting number of images:', len(segmapListFunc))
    shotNoise_x00percent = 3
    shotNoise = iaa.imgcorruptlike.ShotNoise(severity=1)
    alteredImageListFunc, alteredMaskListFunc = expandList(baseImageListFunc, baseMaskListFunc, shotNoise_x00percent)
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

    multiplyAug_x00percent = 3
    multiplyAug = iaa.Multiply((0.3, 1.8))
    alteredImageListFunc, alteredMaskListFunc = expandList(baseImageListFunc, baseMaskListFunc, multiplyAug_x00percent)
    (alteredImageListFunc, alteredMaskListFunc) = multiplyAug(images=alteredImageListFunc,
                                                              segmentation_maps=alteredMaskListFunc)

    fullImageListFunc.extend(alteredImageListFunc)
    segmapListFunc.extend(alteredMaskListFunc)
    return fullImageListFunc, segmapListFunc


def dictLinearContrast(baseImageListFunc, baseMaskListFunc, fullImageListFunc, segmapListFunc):
    print('Contrast, starting number of images:', len(segmapListFunc))
    linearContrast_x00percent = 3
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


root = Tk()
root.withdraw()
binaryMaskFileNames = getFilesInFolderList("Select Binary Mask Image Folder", ".png")
rawImageFileNames = getFilesInFolderList("Select Raw Image Folder", ".jpg")

if len(binaryMaskFileNames) != len(rawImageFileNames):
    print(
        "There are a different number of files in each of the selected folders. Make sure you have a masked image for each raw image!")
    quit()

# Still don't think we should need 256 colors, but this appears to work well
# Actually, this might be because the mask areas are indicated by a value of 255 not 1, so it is assigning it to a class number 255
colorList = [(255, 255, 255) for i in range(256)]
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
    print("Processing image number", entry)
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
    fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "TranslateX")
    fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "TranslateY")
    fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "GaussianBlur")
    fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "ScalingAug")
    fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "ShotNoise")
    fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "ShearX")
    fullImageList, segmapList = imageAugment(baseImageList, baseMaskList, fullImageList, segmapList, "ShearY")

    augmentedRawImagesDir = 'AugmentedRawImages'
    if not os.path.isdir(augmentedRawImagesDir):
        os.mkdir(augmentedRawImagesDir)

    augmentedMaskedImagesDir = 'AugmentedMaskImages'
    if not os.path.isdir(augmentedMaskedImagesDir):
        os.mkdir(augmentedMaskedImagesDir)

    plotList = []
    progressBar = ProgressBar(total=len(fullImageList) - 1)
    for i in range(len(fullImageList)):
        if showPlots:
            plotList.append(fullImageList[i])
            plotList.append(segmapList[i].draw_on_image(fullImageList[i], colors=colorList)[0])
        if saveFiles:
            imageio.imwrite(os.path.join(augmentedRawImagesDir, baseFileName + '_' + str(i) + '.' + baseFileType),
                            fullImageList[i])
            imageio.imwrite(
                os.path.join(augmentedMaskedImagesDir, baseFileName + '_mask_' + str(i) + '.' + maskFileType),
                segmapList[i].draw(colors=colorList)[0])
        progressBar.print_progress_bar(i)

    if showPlots:
        grid_image = ia.draw_grid(plotList)
        ia.imshow(grid_image)
        if saveFiles:
            imageio.imwrite("Grid.jpg", grid_image)
