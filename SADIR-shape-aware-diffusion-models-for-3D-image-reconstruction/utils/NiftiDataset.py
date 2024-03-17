import SimpleITK as sitk
import tensorflow as tf
import os
import re
import numpy as np
import math
import random
from skimage import exposure
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as interpolation
from sklearn.model_selection import train_test_split
import scipy


# ------- Swithes -------

interpolator_image = sitk.sitkBSpline                  # interpolator image
interpolator_label = sitk.sitkBSpline                  # interpolator label

_interpolator_image = 'bspline'          # interpolator image
_interpolator_label = 'bspline'          # interpolator label

Segmentation = True

# ------------------------------------- Functions ---------------------------------------
def write_list(name, my_list):
    with open(name, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)


def split_train_set(split_size, images, labels):

    mapIndexPosition = list(zip(images, labels))  # shuffle order list
    random.shuffle(mapIndexPosition)
    images, labels = zip(*mapIndexPosition)

    images, test_images, labels, test_labels = train_test_split(images, labels, test_size=split_size, random_state=42)

    images = sorted(images, key=numericalSort)
    labels = sorted(labels, key=numericalSort)

    test_images = sorted(test_images, key=numericalSort)
    test_labels = sorted(test_labels, key=numericalSort)

    return images, test_images, labels, test_labels


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def lstFiles(Path):

    images_list = []  # create an empty list, the raw image data files is stored here
    for dirName, subdirList, fileList in os.walk(Path):
        for filename in fileList:
            if ".nii.gz" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".nii" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".nrrd" in filename.lower():
                images_list.append(os.path.join(dirName, filename))

    images_list = sorted(images_list, key=numericalSort)
    return images_list


def resize(img, new_size, interpolator):
    # img = sitk.ReadImage(img)
    dimension = img.GetDimension()

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)

    reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                  zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

    # Create the reference image with a zero origin, identity direction cosine matrix and dimension
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()
    reference_size = new_size
    reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]

    reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as
    # this takes into account size, spacing and direction cosines. For the vast majority of images the direction
    # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the
    # spacing will not yield the correct coordinates resulting in a long debugging session.
    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    # Transform which maps from the reference_image to the current img with the translation mapping the image
    # origins to each other.
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)
    # Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth
    # segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that
    # no new labels are introduced.

    return sitk.Resample(img, reference_image, centered_transform, interpolator, 0.0)


def resample_sitk_image(sitk_image, spacing=None, interpolator=None, fill_value=0):
    # https://github.com/SimpleITK/SlicerSimpleFilters/blob/master/SimpleFilters/SimpleFilters.py
    _SITK_INTERPOLATOR_DICT = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'gaussian': sitk.sitkGaussian,
        'label_gaussian': sitk.sitkLabelGaussian,
        'bspline': sitk.sitkBSpline,
        'hamming_sinc': sitk.sitkHammingWindowedSinc,
        'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
        'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
        'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
    }

    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.
    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    spacing : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int
    Returns
    -------
    SimpleITK image.
    """

    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()

    if not interpolator:
        interpolator = 'linear'
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                'Set `interpolator` manually, '
                'can only infer for 8-bit unsigned or 16, 32-bit signed integers')
        if pixelid == 1:  # 8-bit unsigned int
            interpolator = 'nearest'

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int)

    if not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing] * num_dim
    else:
        new_spacing = [float(s) for s in spacing]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(), \
        '`interpolator` should be one of {}'.format(_SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    new_size = orig_size * (orig_spacing / new_spacing)
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]  # SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()

    resampled_sitk_image = resample_filter.Execute(sitk_image,
                                                   new_size,
                                                   sitk.Transform(),
                                                   sitk_interpolator,
                                                   orig_origin,
                                                   new_spacing,
                                                   orig_direction,
                                                   fill_value,
                                                   orig_pixelid)

    return resampled_sitk_image


def matrix_from_axis_angle(a):
    ux, uy, uz, theta = a
    c = np.cos(theta)
    s = np.sin(theta)
    ci = 1.0 - c
    R = np.array([[ci * ux * ux + c,
                   ci * ux * uy - uz * s,
                   ci * ux * uz + uy * s],
                  [ci * uy * ux + uz * s,
                   ci * uy * uy + c,
                   ci * uy * uz - ux * s],
                  [ci * uz * ux - uy * s,
                   ci * uz * uy + ux * s,
                   ci * uz * uz + c],
                  ])
    return R


def resample_image(image, transform):
    reference_image = image
    interpolator = interpolator_image
    default_value = 0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def resample_label(image, transform):
    reference_image = image
    interpolator = interpolator_label
    default_value = 0
    return sitk.Resample(image, reference_image, transform, interpolator, default_value)


def get_center(img):
    width, height, depth = img.GetSize()
    return img.TransformIndexToPhysicalPoint((int(np.ceil(width / 2)),
                                              int(np.ceil(height / 2)),
                                              int(np.ceil(depth / 2))))


def rotation3d_image(image, theta_x, theta_y, theta_z):
    """
    This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
    respectively
    :param image: An sitk MRI image
    :param theta_x: The amount of degrees the user wants the image rotated around the x axis
    :param theta_y: The amount of degrees the user wants the image rotated around the y axis
    :param theta_z: The amount of degrees the user wants the image rotated around the z axis
    :param show: Boolean, whether or not the user wants to see the result of the rotation
    :return: The rotated image
    """
    theta_x = np.deg2rad(theta_x)
    theta_y = np.deg2rad(theta_y)
    theta_z = np.deg2rad(theta_z)
    euler_transform = sitk.Euler3DTransform(get_center(image), theta_x, theta_y, theta_z, (0, 0, 0))
    image_center = get_center(image)
    euler_transform.SetCenter(image_center)
    euler_transform.SetRotation(theta_x, theta_y, theta_z)
    resampled_image = resample_image(image, euler_transform)
    return resampled_image


def rotation3d_label(image, theta_x, theta_y, theta_z):
   """
   This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
   respectively
   :param image: An sitk MRI image
   :param theta_x: The amount of degrees the user wants the image rotated around the x axis
   :param theta_y: The amount of degrees the user wants the image rotated around the y axis
   :param theta_z: The amount of degrees the user wants the image rotated around the z axis
   :param show: Boolean, whether or not the user wants to see the result of the rotation
   :return: The rotated image
   """
   theta_x = np.deg2rad(theta_x)
   theta_y = np.deg2rad(theta_y)
   theta_z = np.deg2rad(theta_z)
   euler_transform = sitk.Euler3DTransform(get_center(image), theta_x, theta_y, theta_z, (0, 0, 0))
   image_center = get_center(image)
   euler_transform.SetCenter(image_center)
   euler_transform.SetRotation(theta_x, theta_y, theta_z)
   resampled_image = resample_label(image, euler_transform)
   return resampled_image


def flipit(image, axes):
    array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()

    if axes == 0:
        array = np.fliplr(array)
    if axes == 1:
        array = np.flipud(array)

    img = sitk.GetImageFromArray(np.transpose(array, axes=(2, 1, 0)))
    img.SetDirection(direction)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)

    return img


def brightness(image):
    array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()

    max = 255
    min = 0

    c = np.random.randint(-20, 20)

    array = array + c

    array[array >= max] = max
    array[array <= 0] = min

    img = sitk.GetImageFromArray(np.transpose(array, axes=(2, 1, 0)))
    img.SetDirection(direction)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)

    return img


def contrast(image):
    array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()

    shape = array.shape
    ntotpixel = shape[0] * shape[1] * shape[2]
    IOD = np.sum(array)
    luminanza = int(IOD / ntotpixel)

    c = np.random.randint(-5, 5)

    d = array - luminanza
    dc = d * abs(c) / 100

    if c >= 0:
        J = array + dc
        J[J >= 255] = 255
        J[J <= 0] = 0
    else:
        J = array - dc
        J[J >= 255] = 255
        J[J <= 0] = 0

    img = sitk.GetImageFromArray(np.transpose(J, axes=(2, 1, 0)))
    img.SetDirection(direction)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)

    return img


def translateit(image, offset, isseg=False):
    order = 0 if isseg == True else 5

    array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()

    array = scipy.ndimage.interpolation.shift(array, (int(offset[0]), int(offset[1]), 0), order=order)

    img = sitk.GetImageFromArray(np.transpose(array, axes=(2, 1, 0)))
    img.SetDirection(direction)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)

    return img


def adapt_eq_histogram(image):

    adapt = sitk.AdaptiveHistogramEqualizationImageFilter()
    adapt.SetAlpha(0.9)
    adapt.SetBeta(0.6)
    image = adapt.Execute(image)  # set mean and std deviation

    normalizeFilter = sitk.NormalizeImageFilter()
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)
    image = normalizeFilter.Execute(image)  # set mean and std deviation
    image = resacleFilter.Execute(image)  # set intensity 0-255

    return image

# --------------------------------------------------------------------------------------

class NiftiDataset(object):
    """
    load image-label pair for training, testing and inference.
    Currently only support linear interpolation method
    Args:
      image_filename (string): Filename of image data.
      label_filename (string): Filename of label data.
      transforms (list): List of SimpleITK image transformations.
      train (bool): Determine whether the dataset class run in training/inference mode. When set to false, an empty label with same metadata as image is generated.
    """

    def __init__(self,
                 image_filename='img.nii',
                 label_filename='label.nii',
                 transforms=None,
                 train=False,
                 test=False):

        # Init membership variables
        self.image_filename = image_filename
        self.label_filename = label_filename
        self.transforms = transforms
        self.train = train
        self.test = test
        self.bit = sitk.sitkFloat32

    def get_dataset(self):

        image_paths = self.image_filename
        label_paths = self.label_filename

        image_paths = image_paths.rstrip()
        label_paths = label_paths.rstrip()

        return self.input_parser(image_paths, label_paths)

    def read_image(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        image = reader.Execute()
        return image

    def input_parser(self, image_path, label_path):  # read image and label
        # read image and label
        image = self.read_image(image_path)

        image = Normalization(image)  # set intensity 0-255

        # cast image and label
        castImageFilter = sitk.CastImageFilter()
        castImageFilter.SetOutputPixelType(self.bit)
        image = castImageFilter.Execute(image)

        if self.train:
            label = self.read_image(label_path)
            if Segmentation is False:
                label = Normalization(label)  # set intensity 0-255
            castImageFilter.SetOutputPixelType(self.bit)
            label = castImageFilter.Execute(label)

        elif self.test:
            label = self.read_image(label_path)
            if Segmentation is False:
                label = Normalization(label)  # set intensity 0-255
            castImageFilter.SetOutputPixelType(self.bit)
            label = castImageFilter.Execute(label)

        else:
            label = sitk.Image(image.GetSize(), self.bit)  # if it´s not train, create a new image with same size and spacing
            if Segmentation is False:
                label = Normalization(label)  # set intensity 0-255
            label.SetOrigin(image.GetOrigin())
            label.SetSpacing(image.GetSpacing())

        sample = {'image': image, 'label': label}

        if self.transforms:  # apply the transforms to image and label (normalization, resampling, patches)
            for transform in self.transforms:
                sample = transform(sample)

        image_np = sitk.GetArrayFromImage(sample['image'])
        label_np = sitk.GetArrayFromImage(sample['label'])

        if Segmentation is True:
            label_np = abs(np.around(label_np))

        # to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])  (actually it´s the contrary)
        image_np = np.transpose(image_np, (2, 1, 0))
        label_np = np.transpose(label_np, (2, 1, 0))

        image_np = image_np[np.newaxis, :, :, :, np.newaxis]
        label_np = label_np[np.newaxis, :, :, :, np.newaxis]

        return image_np, label_np  # this is the final output to feed the network


def Normalization(image):
    """
    Normalize an image to 0 - 255 (8bits)
    """
    normalizeFilter = sitk.NormalizeImageFilter()
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)

    image = normalizeFilter.Execute(image)  # set mean and std deviation
    image = resacleFilter.Execute(image)  # set intensity 0-255

    return image


class StatisticalNormalization(object):
    """
    Normalize an image by mapping intensity with intensity distribution
    """

    def __init__(self, sigma):
        self.name = 'StatisticalNormalization'
        assert isinstance(sigma, float)
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        statisticsFilter = sitk.StatisticsImageFilter()
        statisticsFilter.Execute(image)

        intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
        intensityWindowingFilter.SetOutputMaximum(255)
        intensityWindowingFilter.SetOutputMinimum(0)
        intensityWindowingFilter.SetWindowMaximum(statisticsFilter.GetMean() + self.sigma * statisticsFilter.GetSigma());
        intensityWindowingFilter.SetWindowMinimum(statisticsFilter.GetMean() - self.sigma * statisticsFilter.GetSigma());

        image = intensityWindowingFilter.Execute(image)

        return {'image': image, 'label': label}


class ManualNormalization(object):
    """
    Normalize an image by mapping intensity with given max and min window level
    """

    def __init__(self, windowMin, windowMax):
        self.name = 'ManualNormalization'
        assert isinstance(windowMax, (int, float))
        assert isinstance(windowMin, (int, float))
        self.windowMax = windowMax
        self.windowMin = windowMin

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
        intensityWindowingFilter.SetOutputMaximum(255)
        intensityWindowingFilter.SetOutputMinimum(0)
        intensityWindowingFilter.SetWindowMaximum(self.windowMax);
        intensityWindowingFilter.SetWindowMinimum(self.windowMin);

        image = intensityWindowingFilter.Execute(image)

        return {'image': image, 'label': label}


class Reorient(object):
    """
    (Beta) Function to orient image in specific axes order
    The elements of the order array must be an permutation of the numbers from 0 to 2.
    """

    def __init__(self, order):
        self.name = 'Reoreient'
        assert isinstance(order, (int, tuple))
        assert len(order) == 3
        self.order = order

    def __call__(self, sample):
        reorientFilter = sitk.PermuteAxesImageFilter()
        reorientFilter.SetOrder(self.order)
        image = reorientFilter.Execute(sample['image'])
        label = reorientFilter.Execute(sample['label'])

        return {'image': image, 'label': label}


class Invert(object):
    """
    Invert the image intensity from 0-255
    """

    def __init__(self):
        self.name = 'Invert'

    def __call__(self, sample):
        invertFilter = sitk.InvertIntensityImageFilter()
        image = invertFilter.Execute(sample['image'], 255)
        label = sample['label']

        return {'image': image, 'label': label}


class Resample(object):
    """
    Resample the volume in a sample to a given voxel size

      Args:
          voxel_size (float or tuple): Desired output size.
          If float, output volume is isotropic.
          If tuple, output voxel size is matched with voxel size
          Currently only support linear interpolation method
    """

    def __init__(self, new_resolution, check):
        self.name = 'Resample'

        assert isinstance(new_resolution, (float, tuple))
        if isinstance(new_resolution, float):
            self.new_resolution = new_resolution
            self.check = check
        else:
            assert len(new_resolution) == 3
            self.new_resolution = new_resolution
            self.check = check

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        new_resolution = self.new_resolution
        check = self.check

        if check is True:
            image = resample_sitk_image(image, spacing=new_resolution, interpolator=_interpolator_image)
            label = resample_sitk_image(label, spacing=new_resolution, interpolator=_interpolator_label)

            return {'image': image, 'label': label}

        if check is False:
            return {'image': image, 'label': label}


class Padding(object):
    """
    Add padding to the image if size is smaller than patch size
      Args:
          output_size (tuple or int): Desired output size. If int, a cubic volume is formed
      """

    def __init__(self, output_size):
        self.name = 'Padding'

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

        assert all(i > 0 for i in list(self.output_size))

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        size_old = image.GetSize()

        if (size_old[0] >= self.output_size[0]) and (size_old[1] >= self.output_size[1]) and (
                size_old[2] >= self.output_size[2]):
            return sample
        else:
            output_size = self.output_size
            output_size = list(output_size)
            if size_old[0] > self.output_size[0]:
                output_size[0] = size_old[0]
            if size_old[1] > self.output_size[1]:
                output_size[1] = size_old[1]
            if size_old[2] > self.output_size[2]:
                output_size[2] = size_old[2]

            output_size = tuple(output_size)

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(image.GetSpacing())
            resampler.SetSize(output_size)

            # resample on image
            resampler.SetInterpolator(sitk.sitkBSpline)
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetOutputDirection(image.GetDirection())
            image = resampler.Execute(image)

            # resample on label
            resampler.SetInterpolator(sitk.sitkBSpline)
            resampler.SetOutputOrigin(label.GetOrigin())
            resampler.SetOutputDirection(label.GetDirection())

            label = resampler.Execute(label)

            return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample. This is usually used for data augmentation.
      Drop ratio is implemented for randomly dropout crops with empty label. (Default to be 0.2)
      This transformation only applicable in train mode
    Args:
      output_size (tuple or int): Desired output size. If int, cubic crop is made.
    """

    def __init__(self, output_size, drop_ratio=0.1, min_pixel=1):
        self.name = 'Random Crop'

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

        assert isinstance(drop_ratio, (int, float))
        if drop_ratio >= 0 and drop_ratio <= 1:
            self.drop_ratio = drop_ratio
        else:
            raise RuntimeError('Drop ratio should be between 0 and 1')

        assert isinstance(min_pixel, int)
        if min_pixel >= 0:
            self.min_pixel = min_pixel
        else:
            raise RuntimeError('Min label pixel count should be integer larger than 0')

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        size_old = image.GetSize()
        size_new = self.output_size

        contain_label = False

        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize([size_new[0], size_new[1], size_new[2]])

        # statFilter = sitk.StatisticsImageFilter()     # not useful
        # statFilter.Execute(label)
        # print(statFilter.GetMaximum(), statFilter.GetSum())

        while not contain_label:
            # get the start crop coordinate in ijk
            if size_old[0] <= size_new[0]:
                start_i = 0
            else:
                start_i = np.random.randint(0, size_old[0] - size_new[0])

            if size_old[1] <= size_new[1]:
                start_j = 0
            else:
                start_j = np.random.randint(0, size_old[1] - size_new[1])

            if size_old[2] <= size_new[2]:
                start_k = 0
            else:
                start_k = np.random.randint(0, size_old[2] - size_new[2])

            roiFilter.SetIndex([start_i, start_j, start_k])

            if Segmentation is False:
                # threshold label into only ones and zero
                threshold = sitk.BinaryThresholdImageFilter()
                threshold.SetLowerThreshold(1)
                threshold.SetUpperThreshold(255)
                threshold.SetInsideValue(1)
                threshold.SetOutsideValue(0)
                mask = threshold.Execute(label)
                mask_cropped = roiFilter.Execute(mask)
                statFilter.Execute(mask_cropped)  # mine for GANs

            if Segmentation is True:
                label_crop = roiFilter.Execute(label)
                statFilter = sitk.StatisticsImageFilter()
                statFilter.Execute(label_crop)

            # will iterate until a sub volume containing label is extracted
            # pixel_count = seg_crop.GetHeight()*seg_crop.GetWidth()*seg_crop.GetDepth()
            # if statFilter.GetSum()/pixel_count<self.min_ratio:
            if statFilter.GetSum() < self.min_pixel:
                contain_label = self.drop(self.drop_ratio)  # has some probabilty to contain patch with empty label
            else:
                contain_label = True

        image_crop = roiFilter.Execute(image)

        return {'image': image_crop, 'label': label_crop}

    def drop(self, probability):
        return random.random() <= probability


class Augmentation(object):
    """
    Application of transforms. This is usually used for data augmentation.
    List of transforms: random noise
    """

    def __init__(self):
        self.name = 'Augmentation'

    def __call__(self, sample):

        choice = np.random.choice([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11])

        # no augmentation
        if choice == 0:  # no augmentation

            image, label = sample['image'], sample['label']
            return {'image': image, 'label': label}

        # Additive Gaussian noise
        if choice == 1:  # Additive Gaussian noise

            mean = np.random.uniform(0, 1)
            std = np.random.uniform(0, 2)
            self.noiseFilter = sitk.AdditiveGaussianNoiseImageFilter()
            self.noiseFilter.SetMean(mean)
            self.noiseFilter.SetStandardDeviation(std)

            image, label = sample['image'], sample['label']
            image = self.noiseFilter.Execute(image)
            if Segmentation is False:
                label = self.noiseFilter.Execute(label)

            return {'image': image, 'label': label}

        # Recursive Gaussian
        if choice == 2:  # Recursive Gaussian

            sigma = np.random.uniform(0, 1.5)
            self.noiseFilter = sitk.RecursiveGaussianImageFilter()
            self.noiseFilter.SetOrder(0)
            self.noiseFilter.SetSigma(sigma)

            image, label = sample['image'], sample['label']
            image = self.noiseFilter.Execute(image)
            if Segmentation is False:
                label = self.noiseFilter.Execute(label)    # comment for segmentation

            return {'image': image, 'label': label}

        # Random rotation x y z
        if choice == 3:  # Random rotation

            theta_x = np.random.randint(-20, 20)
            theta_y = np.random.randint(-20, 20)
            theta_z = np.random.randint(-180, 180)
            image, label = sample['image'], sample['label']

            image = rotation3d_image(image,theta_x,theta_y, theta_z)
            label = rotation3d_label(label,theta_x,theta_y, theta_z)

            return {'image': image, 'label': label}

        # BSpline Deformation
        if choice == 4:  # BSpline Deformation

            randomness = 10

            assert isinstance(randomness, (int, float))
            if randomness > 0:
                self.randomness = randomness
            else:
                raise RuntimeError('Randomness should be non zero values')

            image, label = sample['image'], sample['label']
            spline_order = 3
            domain_physical_dimensions = [image.GetSize()[0] * image.GetSpacing()[0],
                                          image.GetSize()[1] * image.GetSpacing()[1],
                                          image.GetSize()[2] * image.GetSpacing()[2]]

            bspline = sitk.BSplineTransform(3, spline_order)
            bspline.SetTransformDomainOrigin(image.GetOrigin())
            bspline.SetTransformDomainDirection(image.GetDirection())
            bspline.SetTransformDomainPhysicalDimensions(domain_physical_dimensions)
            bspline.SetTransformDomainMeshSize((10, 10, 10))

            # Random displacement of the control points.
            originalControlPointDisplacements = np.random.random(len(bspline.GetParameters())) * self.randomness
            bspline.SetParameters(originalControlPointDisplacements)

            image = sitk.Resample(image, bspline)
            label = sitk.Resample(label, bspline)
            return {'image': image, 'label': label}

        # Random flip
        if choice == 5:  # Random flip

            axes = np.random.choice([0, 1])
            image, label = sample['image'], sample['label']

            image = flipit(image, axes)
            label = flipit(label, axes)

            return {'image': image, 'label': label}

        # Brightness
        if choice == 6:  # Brightness

            image, label = sample['image'], sample['label']

            image = brightness(image)
            if Segmentation is False:
                label = brightness(label)

            return {'image': image, 'label': label}

        # Contrast
        if choice == 7:  # Contrast

            image, label = sample['image'], sample['label']

            image = contrast(image)
            if Segmentation is False:
                label = contrast(label)             # comment for segmentation

            return {'image': image, 'label': label}

        # Translate
        if choice == 8:  # translate

            image, label = sample['image'], sample['label']

            t1 = np.random.randint(-30, 30)
            t2 = np.random.randint(-30, 30)
            offset = [t1, t2]

            image = translateit(image, offset)
            label = translateit(label, offset)

            return {'image': image, 'label': label}

        # Random rotation z
        if choice == 9:  # Random rotation

            theta_x = 0
            theta_y = 0
            theta_z = np.random.randint(-180, 180)
            image, label = sample['image'], sample['label']

            image = rotation3d_image(image, theta_x, theta_y, theta_z)
            label = rotation3d_label(label, theta_x, theta_y, theta_z)

            return {'image': image, 'label': label}

        # Random rotation x
        if choice == 10:  # Random rotation

            theta_x = np.random.randint(-20, 20)
            theta_y = 0
            theta_z = 0
            image, label = sample['image'], sample['label']

            image = rotation3d_image(image, theta_x, theta_y, theta_z)
            label = rotation3d_label(label, theta_x, theta_y, theta_z)

            return {'image': image, 'label': label}

        # Random rotation y
        if choice == 11:  # Random rotation

            theta_x = 0
            theta_y = np.random.randint(-20, 20)
            theta_z = 0
            image, label = sample['image'], sample['label']

            image = rotation3d_image(image, theta_x, theta_y, theta_z)
            label = rotation3d_label(label, theta_x, theta_y, theta_z)

            return {'image': image, 'label': label}

        # # histogram adaptequ
        # if choice == 9:
        #
        #     image, label = sample['image'], sample['label']
        #
        #     image = adapt_eq_histogram(image)
        #
        #     return {'image': image, 'label': label}


class ConfidenceCrop(object):
    """
    Crop the image in a sample that is certain distance from individual labels center.
    This is usually used for data augmentation with very small label volumes.
    The distance offset from connected label centroid is model by Gaussian distribution with mean zero and user input sigma (default to be 2.5)
    i.e. If n isolated labels are found, one of the label's centroid will be randomly selected, and the cropping zone will be offset by following scheme:
    s_i = np.random.normal(mu, sigma*crop_size/2), 1000)
    offset_i = random.choice(s_i)
    where i represents axis direction
    A higher sigma value will provide a higher offset
    Args:
      output_size (tuple or int): Desired output size. If int, cubic crop is made.
      sigma (float): Normalized standard deviation value.
    """

    def __init__(self, output_size, sigma=2.5):
        self.name = 'Confidence Crop'

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

        assert isinstance(sigma, (float, tuple))
        if isinstance(sigma, float) and sigma >= 0:
            self.sigma = (sigma, sigma, sigma)
        else:
            assert len(sigma) == 3
            self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        size_new = self.output_size

        # guarantee label type to be integer
        castFilter = sitk.CastImageFilter()
        castFilter.SetOutputPixelType(sitk.sitkUInt8)
        label = castFilter.Execute(label)

        ccFilter = sitk.ConnectedComponentImageFilter()
        labelCC = ccFilter.Execute(label)

        labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
        labelShapeFilter.Execute(labelCC)

        if labelShapeFilter.GetNumberOfLabels() == 0:
            # handle image without label
            selectedLabel = 0
            centroid = (int(self.output_size[0] / 2), int(self.output_size[1] / 2), int(self.output_size[2] / 2))
        else:
            # randomly select of the label's centroid
            selectedLabel = random.randint(1, labelShapeFilter.GetNumberOfLabels())
            centroid = label.TransformPhysicalPointToIndex(labelShapeFilter.GetCentroid(selectedLabel))

        centroid = list(centroid)

        start = [-1, -1, -1]  # placeholder for start point array
        end = [self.output_size[0] - 1, self.output_size[1] - 1,
               self.output_size[2] - 1]  # placeholder for start point array
        offset = [-1, -1, -1]  # placeholder for start point array
        for i in range(3):
            # edge case
            if centroid[i] < (self.output_size[i] / 2):
                centroid[i] = int(self.output_size[i] / 2)
            elif (image.GetSize()[i] - centroid[i]) < (self.output_size[i] / 2):
                centroid[i] = image.GetSize()[i] - int(self.output_size[i] / 2) - 1

            # get start point
            while ((start[i] < 0) or (end[i] > (image.GetSize()[i] - 1))):
                offset[i] = self.NormalOffset(self.output_size[i], self.sigma[i])
                start[i] = centroid[i] + offset[i] - int(self.output_size[i] / 2)
                end[i] = start[i] + self.output_size[i] - 1

        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize(self.output_size)
        roiFilter.SetIndex(start)
        croppedImage = roiFilter.Execute(image)
        croppedLabel = roiFilter.Execute(label)

        return {'image': croppedImage, 'label': croppedLabel}

    def NormalOffset(self, size, sigma):
        s = np.random.normal(0, size * sigma / 2, 100)  # 100 sample is good enough
        return int(round(random.choice(s)))


class BSplineDeformation(object):
    """
    Image deformation with a sparse set of control points to control a free form deformation.
    Details can be found here:
    https://simpleitk.github.io/SPIE2018_COURSE/spatial_transformations.pdf
    https://itk.org/Doxygen/html/classitk_1_1BSplineTransform.html
    Args:
      randomness (int,float): BSpline deformation scaling factor, default is 4.
    """

    def __init__(self, randomness=4):
        self.name = 'BSpline Deformation'

        assert isinstance(randomness, (int, float))
        if randomness > 0:
            self.randomness = randomness
        else:
            raise RuntimeError('Randomness should be non zero values')

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        spline_order = 3
        domain_physical_dimensions = [image.GetSize()[0] * image.GetSpacing()[0],
                                      image.GetSize()[1] * image.GetSpacing()[1],
                                      image.GetSize()[2] * image.GetSpacing()[2]]

        bspline = sitk.BSplineTransform(3, spline_order)
        bspline.SetTransformDomainOrigin(image.GetOrigin())
        bspline.SetTransformDomainDirection(image.GetDirection())
        bspline.SetTransformDomainPhysicalDimensions(domain_physical_dimensions)
        bspline.SetTransformDomainMeshSize((4, 4, 4))

        # Random displacement of the control points.
        originalControlPointDisplacements = np.random.random(len(bspline.GetParameters())) * self.randomness
        bspline.SetParameters(originalControlPointDisplacements)

        image = sitk.Resample(image, bspline)
        label = sitk.Resample(label, bspline)
        return {'image': image, 'label': label}

    def NormalOffset(self, size, sigma):
        s = np.random.normal(0, size * sigma / 2, 100)  # 100 sample is good enough
        return int(round(random.choice(s)))
