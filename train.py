import os
import argparse
import csv
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from doa_math import DoaClasses, lookup_class_index


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_entries, loss, batch_size=64, dim=(32,32,32), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data_entries = data_entries
        self.loss_type = loss
        self.classes = DoaClasses()
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_entries) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation([self.data_entries[k] for k in indexes])

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_entries))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_entries):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        if  self.loss_type == 'categorical':
            y = np.empty((self.batch_size, 25), dtype=int)
        elif self.loss_type == 'cartesian':
            y = np.empty((self.batch_size, 25, 3), dtype=np.float32)

        # Generate data
        for i, entry in enumerate(batch_entries):
            try:
                X[i,] = np.load(entry[0])
            except Exception as e:
                print(str(e))
                print("Error loading: " + str(i))\

            if self.loss_type == 'categorical':
                # we'll keep training Laureline's model in their coordinate convention and do another conversion at output just to make life easier
                raw_label = np.array(entry[1][3:5]).astype("float32")
                label = lookup_class_index(np.pi/2-raw_label[1], raw_label[0], self.classes)
                y[i,] = np.array([label]*25)
            elif self.loss_type == 'cartesian':
                raw_label = np.array(entry[1][0:3]).astype("float32")
                y[i,] = np.array([raw_label]*25)

        if self.loss_type == 'cartesian':
            return X, y
        else:
            return X, keras.utils.to_categorical(y, num_classes=len(self.classes.classes))


def read_data_entries(labelpath, data_folder):
    # initialize dataset
    csvfile = open(labelpath, 'r')
    csv_reader = csv.reader(csvfile, delimiter=',')
    next(csv_reader, None)
    data_entries = []
    for line in csv_reader:
        npypath = os.path.join(data_folder, line[0])
        if not npypath.endswith('npy'):
            npypath = npypath + '.npy'
        if os.path.exists(npypath):
            data_entries.append([npypath, [float(x) for x in line[1:]]])
    return data_entries


def main():
    parser = argparse.ArgumentParser(prog='train',
                                     description="""Script to train a DOA estimator""")
    parser.add_argument("--input", "-i", required=True, help="Directory where data and labels are", type=str)
    parser.add_argument("--label", "-l", required=True, help="Path to the label csv", type=str)
    parser.add_argument("--output", "-o", default="models", help="Directory to write results", type=str)
    parser.add_argument("--batchsize", "-b", type=int, default=256, help="Choose a batchsize")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path")
    parser.add_argument("--loss", "-lo", type=str, choices=["categorical", "cartesian"],
                        required=True, help="Choose loss representation")

    args = parser.parse_args()
    assert os.path.exists(args.input), "Input folder does not exist!"

    epochs = args.epochs
    batchsize = args.batchsize
    label_path = args.label
    foldername = '{}_batch{}'.format(args.loss, batchsize)

    outpath = os.path.join(args.output, foldername)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    savepath = os.path.join(outpath, 'best_model.{epoch:02d}-{val_loss:.6f}.h5')

    assert os.path.exists(label_path), "Label csv does not exist!"
    train_data = read_data_entries(label_path, args.input)
    train_data_entries, val_data_entries = train_test_split(train_data, test_size=0.3, random_state=11)
    
    # Parameters
    params = {'dim': (25, 513, 6),
            'batch_size': batchsize,
            'loss': args.loss,
            'shuffle': True}

    # Generators
    training_generator = DataGenerator(train_data_entries, **params)
    validation_generator = DataGenerator(val_data_entries, **params)

    model = load_model(args.model)
    model.summary()


    # Train model on dataset
    earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(savepath, verbose=1, save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs,
                                  verbose=1, callbacks=[earlyStopping, mcp_save])

    xs = history.epoch

    # Plot training & validation loss values
    plt.plot(xs, history.history['loss'])
    plt.plot(xs, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    plt.savefig(os.path.join(outpath, 'train_val_loss_curve.png'))


if __name__ == "__main__":
    main()
