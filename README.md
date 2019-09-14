# doa-release
Keras (w/ tensorflow backend) code accompanying our paper [Regression and Classification for Direction-of-Arrival Estimation with Convolutional Recurrent Neural Networks](https://arxiv.org/abs/1904.08452), Interspeech 2019

## Dataset

As described in the paper, our training/validation set is purely simulated data using a geometric sound propagation engine, whereas the test set only consists of real-world recorded data. To train/test our models, you need to download following files:

- Training/validation set features and labels [33.4GB] https://obj.umiacs.umd.edu/gammadata/dataset/doa/features.zip
- Test set original wav files and labels [19.2GB] https://obj.umiacs.umd.edu/gammadata/dataset/doa/SOFA_DOA_test_set.zip
- Training/validation set original wav files (you can skip this one if you don't want to extract the feature yourself) [43.8GB] https://obj.umiacs.umd.edu/gammadata/dataset/doa/wav.zip

The reason for providing test set in wav format instead of feature is that our models are trained assuming the ACN convention, whereas others may assume FuMa convention (see [Ambisonic formats](https://en.wikipedia.org/wiki/Ambisonic_data_exchange_formats)). In the test code, we provide a simple function for format conversion before feature extraction if needed.

## Models

In this repo, we attach several models under `models` folder. These are:

| Model name                           | Explanation                                                                                    | Convention |
|--------------------------------------|------------------------------------------------------------------------------------------------|------------|
| cartesian_base_model.h5              | Initial (untrained) regression model                                                           |     ACN    |
| cartesian_trained_model.h5           | Trained regression model                                                                       | ACN        |
| categorical_trained_model.h5         | Trained classification model                                                                   | ACN        |
| Perotin_categorical_trained_model.h5 | Baseline classification model by [Perontin et al.](https://hal.inria.fr/hal-01840453/document) | FuMa       |
Note that `categorical_trained_model.h5` is trained from `Perotin_categorical_trained_model.h5`, which is not different from training from scratch because their model uses FuMa convention that generates very different features from ACN convention.

## Usage

0. Download dataset and clone this repo.
1. Build conda environment and activate it:
    ```
    conda env create --file=environment.yml 
    conda activate doa-release
    ```
2. Training
    ```
    python3 train.py -i [train_feature_dir] -l [train_feature_dir]/train_labels.csv -o [output_dir] -lo [cartesian or categorical] -m models/[model_name]
    ```
3. Testing
    ```
    python3 test.py -i [test_wav_dir] -l [test_wav_dir]/test_labels.csv -m [trained_model_path] -lo [cartesian or categorical]
    ```
    When testing a model that uses FuMa convention, you must append `-c` to test.py's argument list, which enables conversion from ACN to FuMa. For example:
    ```
    python3 test.py -i [test_wav_dir] -l [test_wav_dir]/test_labels.csv -m models/Perotin_categorical_trained_model.h5 -lo categorical -c
    ```

## Citation
If you use our codes or models, please consider citing:
```
@inproceedings{tang2019regression,
  title={Regression and Classification for Direction-of-Arrival Estimation with Convolutional Recurrent Neural Networks},
  author={Tang, Zhenyu and Kanu, John.D and Hogan, Kevin and Manocha, Dinesh},
  booktitle={Interspeech},
  year={2019},
}
```
We also recommend citing the work of Perontin et al. if you also use their model:
```
@inproceedings{perotin2018crnn,
  title={CRNN-based joint azimuth and elevation localization with the Ambisonics intensity vector},
  author={Perotin, Laur{\'e}line and Serizel, Romain and Vincent, Emmanuel and Gu{\'e}rin, Alexandre},
  booktitle={2018 16th International Workshop on Acoustic Signal Enhancement (IWAENC)},
  pages={241--245},
  year={2018},
  organization={IEEE}
}
```
