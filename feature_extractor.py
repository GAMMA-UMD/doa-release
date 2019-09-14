import os
import numpy as np
import argparse
from time import time
from multiprocessing import Pool
from scipy.io.wavfile import read
import dsp

overlapping = 512
chunk_size = 1024
num_frames = 25


def getNormalizedIntensity(x_f):
    '''
    Compute the input feature needed by the localization neural network.

    Parameters
    ----------
    x_f: nd-array
        STFT of the HOA signal.
        Shape: (nBatch, nChannel, lSentence, nBand)

    Returns
    -------
    inputFeat_f: nd-array
        Shape: (nBatch, lSentence, nBand, nFeat)
    '''
    (lBatch, nChannel, lSentence, nBand) = x_f.shape
    nFeat = 6
    inputFeat_f = np.empty((lBatch, lSentence, nBand, nFeat), dtype=np.float32)

    for nBatch, sig_f in enumerate(x_f):  # Process all examples in the batch
        # Compute the intensity vector in each TF bin
        intensityVect = np.empty((lSentence, nBand, 3), dtype=complex)
        intensityVect[:, :, 0] = sig_f[0].conj() * sig_f[1]
        intensityVect[:, :, 1] = sig_f[0].conj() * sig_f[2]
        intensityVect[:, :, 2] = sig_f[0].conj() * sig_f[3]

        # Normalize it in each TF bin
        coeffNorm = (abs(sig_f[0]) ** 2 + np.sum(abs(sig_f[1:]) ** 2 / 3, axis=0))[:, :, np.newaxis]
        inputFeat_f[nBatch, :, :, :nFeat // 2] = np.real(intensityVect) / coeffNorm
        inputFeat_f[nBatch, :, :, nFeat // 2:] = np.imag(intensityVect) / coeffNorm

    return inputFeat_f


def save_feature(filepath, savepath, mapping=None):
    fs, x = read(filepath)
    assert fs == 16000
    x = x.T
    if mapping is not None:
        newx = np.empty_like(x)
        mapping = np.insert(mapping, 0, 0)
        for i in range(len(mapping)):
            sign = np.sign(mapping[i]) if i > 0 else 1
            newx[i, :] = sign * x[abs(mapping[i]), :]
        x = newx

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
    inputFeat_f = getNormalizedIntensity(x_f)
    np.save(savepath, inputFeat_f)


def main():
    parser = argparse.ArgumentParser(prog='feature_extractor',
                                     description="""Script to convert ambisonic audio to intensity vectors""")
    parser.add_argument("--audiodir", "-d", help="Directory where audio files are located",
                        type=str, required=True)
    parser.add_argument("--output", "-o", help="Directory where feature files are written to",
                        type=str, required=True)
    parser.add_argument("--nthreads", "-n", type=int, default=1, help="Number of threads to use")

    args = parser.parse_args()
    audiodir = args.audiodir
    nthreads = args.nthreads
    outpath = args.output

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    ts = time()
    # Convert all houses
    try:
        # # Create a pool to communicate with the worker threads
        pool = Pool(processes=nthreads)
        for subdir, dirs, files in os.walk(audiodir):
            for f in files:
                if f.endswith('.wav'):
                    filename = os.path.join(subdir, f)
                    savepath = os.path.join(outpath, f.replace('.wav', '.npy'))
                    pool.apply_async(save_feature, args=(filename, savepath))
    except Exception as e:
        print(str(e))
        pool.close()
    pool.close()
    pool.join()
    print('Took {}'.format(time() - ts))


if __name__ == "__main__":
    main()
