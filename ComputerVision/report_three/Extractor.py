import cv2
from math import copysign, log10
import numpy as np
from skimage import feature
import mahotas as mt


class Extractor:

    @staticmethod
    def hu_extraction(data):
        new_data = []

        for im in data:
            im = im.reshape(35, 35)
            im = im.astype(np.uint8)

            # Calculate Moments
            moments = cv2.moments(im)

            # Calculate Hu Moments
            hu_moments = cv2.HuMoments(moments)

            # Log scale hu moments
            hu = [-1 * copysign(1.0, hu_moment) * log10(abs(hu_moment)) if hu_moment != 0 else 0 for hu_moment in hu_moments]
            new_data.append(hu)

        return np.array(new_data)

    @staticmethod
    def lbp_extraction(data, numPoints, radius):
        new_data = []

        for im in data:
            im = im.reshape(35, 35)
            im = im.astype(np.uint8)

            hist = Extractor.lbp_describe(im, numPoints, radius)
            new_data.append(hist)

        return np.array(new_data)

    @staticmethod
    def lbp_describe(image, num_points, radius, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, num_points,
                                           radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, num_points + 3),
                                 range=(0, num_points + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist

    @staticmethod
    def glcm_extraction(data):
        new_data = []

        for im in data:
            im = im.reshape(35, 35)
            im = im.astype(np.uint8)

            g = feature.greycomatrix(im, [0, 1], [0, np.pi/2], levels=8, normed=True, symmetric=True)
            contrast = feature.greycoprops(g, 'contrast').flatten()
            energy = feature.greycoprops(g, 'energy').flatten()
            homogeneity = feature.greycoprops(g, 'homogeneity').flatten()
            correlation = feature.greycoprops(g, 'correlation').flatten()
            dissimilarity = feature.greycoprops(g, 'dissimilarity').flatten()
            asm = feature.greycoprops(g, 'ASM').flatten()

            features = np.concatenate((contrast, energy, homogeneity, correlation, dissimilarity, asm))
            new_data.append(features)

        return np.array(new_data)

    @staticmethod
    def extract_features(image):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean

