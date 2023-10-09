from __future__ import division, print_function
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset


def data_generator(images_list, labels_list, batch_size, Transforms):

    f = open(images_list, 'r')
    images = f.readlines()
    f.close()

    f = open(labels_list, 'r')
    labels = f.readlines()
    f.close()

    c = 0

    while (True):

        mapIndexPosition = list(zip(images, labels))  # shuffle order list
        random.shuffle(mapIndexPosition)
        images, labels = zip(*mapIndexPosition)

        for i in range(c, c + batch_size):

            TrainDataset = NiftiDataset.NiftiDataset(
                image_filename=images[i],
                label_filename=labels[i],
                transforms=Transforms,
                train=True
            )

            trainDataset = TrainDataset.get_dataset()
            # trainDataset = trainDataset.batch(batch_size)

        c += batch_size

        if c + batch_size >= len(images):
            c = 0

        yield (trainDataset)


