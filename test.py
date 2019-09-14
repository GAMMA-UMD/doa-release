# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:19:30 2019

@author: WSJF7149
"""
import csv
import os
import argparse
import math
import grid
import dsp
import numpy as np
from scipy.io.wavfile import read
from keras.models import load_model
from feature_extractor import getNormalizedIntensity


def toLOCATA(azi, ele):
    # azi -= 90
    # if (azi < -180):
    #     azi += 360
    ele = 90 - ele
    return azi, ele


def tensor_angle(a, b):
    half_delta = (a - b) / 2
    temp = np.clip(math.sin(half_delta[..., 1]) ** 2 + math.sin(a[..., 1]) * math.sin(b[..., 1]) * math.sin(
        half_delta[..., 0]) ** 2, 1e-9, None)
    angle = 2 * math.asin(np.clip(math.sqrt(temp), None, 1.0 - 1e-9))
    num_nan = np.sum(np.isnan(angle))
    if num_nan > 0:
        print("encountered {} NANs".format(num_nan))
    return angle


def acn2fuma(x_in):
    x_out = np.array([x_in[0, :] / math.sqrt(2.0), x_in[3, :], x_in[1, :], x_in[2, :]])
    return x_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test',
                                     description="""Script to test the DOA estimation system""")
    parser.add_argument("--input", "-i", required=True, help="Directory of input audio files", type=str)
    parser.add_argument("--label", "-l", required=True, help="Path to label", type=str)
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path")
    parser.add_argument("--loss", "-lo", type=str, choices=["categorical", "cartesian"],
                        default="categorical", help="Choose loss representation")
    parser.add_argument('--convert', "-c", dest='do_convert', action='store_true',
                        help='flag to enable ACN to FUMA conversion for FOA channel ordering')
    parser.set_defaults(do_convert=False)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("input path {} non-exist, abort!".format(args.input))
        exit(1)

    labelpath = os.path.join(args.label)
    csvfile = open(labelpath, 'r')
    csvreader = csv.reader(csvfile)
    next(csvreader, None)
    test_data = list(csvreader)

    model = load_model(args.model)

    feature_tensor = []
    labels = []
    for line in test_data:
        # read data and label
        wavpath = os.path.join(args.input, line[0])
        if not wavpath.endswith('wav'):
            wavpath = wavpath + '.wav'
        fs, x = read(wavpath)
        x = x.T
        if args.do_convert:
            print("converting ACN input to FUMA...")
            x = acn2fuma(x)
        # Compute the STFT
        nChannel, nSmp = x.shape
        lFrame = 1024  # size of the STFT window, in samples
        nBand = lFrame // 2 + 1
        nOverlap = lFrame // 2
        hop = lFrame - nOverlap
        lSentence = nSmp // hop
        x_f = np.empty((nChannel, lSentence, nBand), dtype=complex)
        for iChannel in range(nChannel):
            x_f[iChannel] = dsp.stft(x[iChannel], lWindow=lFrame)

        # The neural network can only process buffers of 25 frames
        lBuffer = 25
        nBatch = 1
        x_f = x_f[np.newaxis, :, :lBuffer, :]  # First axis corresponding to the batches of 25 frames to process; here there's only 1

        # Get the input feature for the neural network
        feature = getNormalizedIntensity(x_f)

        label = [float(x) for x in line[4:6]]
        labels.append(label)
        feature_tensor.append(feature)

    inputFeat_f = np.vstack(feature_tensor)
    nBatch = inputFeat_f.shape[0]
    print("feature:{}".format(inputFeat_f.shape))

    # Make the prediction
    predictProbaByFrame = model.predict(inputFeat_f)

    # Average the predictions over a sequence of 25 frames
    predictProbaByBatch = np.mean(predictProbaByFrame, axis=1)

    # Load the discretized sphere corresponding to the network's output classes
    gridStep = 10  # Approximate angle between two DoAs on the grid, in degrees
    neighbour_tol = 2 * gridStep  # angular diameter of a neighborhood, for spatial smoothing and peak selection
    el_grid, az_grid = grid.makeDoaGrid(gridStep)

    # Get the main peaks and the corresponding elevations and azimuths
    el_pred = np.empty(nBatch)
    az_pred = np.empty(nBatch)

    errors = []
    for iBatch in range(nBatch):
        if args.loss == "cartesian":
            dir = predictProbaByBatch[iBatch] / np.linalg.norm(predictProbaByBatch[iBatch])
            new_azi = math.atan2(-dir[1], -dir[0])
            new_ele = math.acos(dir[2])
            error = tensor_angle(np.array([new_azi, new_ele]), np.array(labels[iBatch]))
            # print("{} vs {}".format(np.array([new_azi, new_ele]), labels[iBatch]))
        else:
            # Find all the main peaks of the prediction (spatial smoothing is performed)
            peaks, iPeaks = grid.peaksOnGrid(predictProbaByBatch[iBatch], el_grid, az_grid, neighbour_tol)

            # Select the right number of sources
            iMax = np.argmax(peaks)
            predIdx = iPeaks[iMax]

            el_pred[iBatch] = el_grid[predIdx]
            az_pred[iBatch] = az_grid[predIdx]

            new_azi, new_ele = toLOCATA(az_pred[iBatch], el_pred[iBatch])
            error = tensor_angle(np.deg2rad([new_azi, new_ele]), np.array(labels[iBatch]))

        errors.append(error)
        print('predicted = ({:.4}, {:.4}), true = ({:.4}, {:.4}), error = {:.4}'.format(np.rad2deg(new_azi),
                                                                                        np.rad2deg(new_ele),
                                                                                        np.rad2deg(labels[iBatch][0]),
                                                                                        np.rad2deg(labels[iBatch][1]),
                                                                                        np.rad2deg(error)))
    angle_observations = np.array([5, 10, 15])
    angle_cnts = np.zeros(shape=angle_observations.shape)
    for i, deg in enumerate(angle_observations):
        angle_cnts[i] += (errors <= np.deg2rad(deg)).sum()
    angle_accuracy = angle_cnts / len(errors)
    for i, accuracy in enumerate(angle_accuracy):
        print("accuracy/deg{}: {:.4}%".format(angle_observations[i], accuracy * 100))
    print("average error: {:.4} degrees".format(np.mean(np.rad2deg(errors))))
    print("{},{:.4}%,{:.4}%,{:.4}%,{:.4}".format(args.model, angle_accuracy[0] * 100, angle_accuracy[1] * 100,
                                                 angle_accuracy[2] * 100, np.mean(np.rad2deg(errors))))
