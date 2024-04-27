#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    PyTorch implementation of a simple 2-layer-deep LSTM for genre classification of musical audio.
    Feeding the LSTM stack are spectral {centroid, contrast}, chromagram & MFCC features (33 total values)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from GenreFeatureData import (
    GenreFeatureData,
)  # local python class with Audio feature extraction (librosa)

from sklearn.metrics import confusion_matrix

# class definition
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=6, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input, hidden=None):
        # lstm step => then ONLY take the sequence's final timetep to pass into the linear/dense layer
        # Note: lstm_out contains outputs for every step of the sequence we are looping over (for BPTT)
        # but we just need the output of the last step of the sequence, aka lstm_out[-1]
        lstm_out, hidden = self.lstm(input, hidden)
        logits = self.linear(lstm_out[-1])              # equivalent to return_sequences=False from Keras
        genre_scores = F.log_softmax(logits, dim=1)
        return genre_scores, hidden

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (
                torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()


def main():
    genre_features = GenreFeatureData()

    # if all of the preprocessed files do not exist, regenerate them all for self-consistency
    if (
            os.path.isfile(genre_features.train_X_preprocessed_data)
            and os.path.isfile(genre_features.train_Y_preprocessed_data)
            and os.path.isfile(genre_features.dev_X_preprocessed_data)
            and os.path.isfile(genre_features.dev_Y_preprocessed_data)
            # and os.path.isfile(genre_features.test_X_preprocessed_data)
            # and os.path.isfile(genre_features.test_Y_preprocessed_data)
    ):
        print("Preprocessed files exist, deserializing npy files")
        genre_features.load_deserialize_data()
    else:
        print("Preprocessing raw audio files")
        genre_features.load_preprocess_data()

    train_X = torch.from_numpy(genre_features.train_X).type(torch.Tensor)
    dev_X = torch.from_numpy(genre_features.dev_X).type(torch.Tensor)
    # test_X = torch.from_numpy(genre_features.test_X).type(torch.Tensor)

    # Targets is a long tensor of size (N,) which tells the true class of the sample.
    train_Y = torch.from_numpy(genre_features.train_Y).type(torch.LongTensor)
    dev_Y = torch.from_numpy(genre_features.dev_Y).type(torch.LongTensor)
    # test_Y = torch.from_numpy(genre_features.test_Y).type(torch.LongTensor)

    # Convert {training, test} torch.Tensors
    print("Training X shape: " + str(genre_features.train_X.shape))
    print("Training Y shape: " + str(genre_features.train_Y.shape))
    print("Validation X shape: " + str(genre_features.dev_X.shape))
    print("Validation Y shape: " + str(genre_features.dev_Y.shape))
    # print("Test X shape: " + str(genre_features.test_X.shape))
    # print("Test Y shape: " + str(genre_features.test_Y.shape))

    savefig = True

    batch_size = 35  # num of training examples per minibatch
    num_epochs = 1000
    learning_rate = 0.00003

    # Define model
    print("Build LSTM RNN model ...")
    model = LSTM(
        input_dim=33, hidden_dim=128, batch_size=batch_size, output_dim=6, num_layers=2
    )
    loss_function = nn.NLLLoss()  # expects ouputs from LogSoftmax

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # To keep LSTM stateful between batches, you can set stateful = True, which is not suggested for training
    stateful = False

    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print("\nTraining on GPU")
    else:
        print("\nNo GPU, training on CPU")

    # all training data (epoch) / batch_size == num_batches (12)
    num_batches = int(train_X.shape[0] / batch_size)
    num_dev_batches = int(dev_X.shape[0] / batch_size)

    train_loss_list, train_accuracy_list, train_epoch_list = [], [], []
    train_lr = []
    val_loss_list, val_accuracy_list, val_epoch_list = [], [], []


    print("Training ...")
    for epoch in range(num_epochs):

        train_lr.append(optimizer.param_groups[0]['lr'])

        train_running_loss, train_acc = 0.0, 0.0

        # Init hidden state - if you don't want a stateful LSTM (between epochs)
        hidden_state = None
        for i in range(num_batches):

            # zero out gradient, so they don't accumulate btw batches
            model.zero_grad()

            # train_X shape: (total # of training examples, sequence_length, input_dim)
            # train_Y shape: (total # of training examples, # output classes)
            #
            # Slice out local minibatches & labels => Note that we *permute* the local minibatch to
            # match the PyTorch expected input tensor format of (sequence_length, batch size, input_dim)
            X_local_minibatch, y_local_minibatch = (
                train_X[i * batch_size: (i + 1) * batch_size, ],
                train_Y[i * batch_size: (i + 1) * batch_size, ],
            )

            # Reshape input & targets to "match" what the loss_function wants
            X_local_minibatch = X_local_minibatch.permute(1, 0, 2)

            # NLLLoss does not expect a one-hot encoded vector as the target, but class indices
            y_local_minibatch = torch.max(y_local_minibatch, 1)[1]

            y_pred, hidden_state = model(X_local_minibatch, hidden_state)  # forward pass

            # Stateful = False for training. Do we go Stateful = True during inference/prediction time?
            if not stateful:
                hidden_state = None
            else:
                h_0, c_0 = hidden_state
                h_0.detach_(), c_0.detach_()
                hidden_state = (h_0, c_0)

            loss = loss_function(y_pred, y_local_minibatch)  # compute loss
            loss.backward()  # backward pass
            optimizer.step()  # parameter update

            train_running_loss += loss.detach().item()  # unpacks the tensor into a scalar value
            train_acc += model.get_accuracy(y_pred, y_local_minibatch)

        train_epoch_list.append(epoch)
        train_accuracy_list.append(train_acc / num_batches)
        train_loss_list.append(train_running_loss / num_batches)

        print(
            "Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f | Lr:"
            % (epoch+1, train_running_loss / num_batches, train_acc / num_batches),
            optimizer.param_groups[0]['lr']
        )

        # Learning rate decay
        if epoch+1 == int(0.3*num_epochs):
            optimizer.param_groups[0]['lr'] = 0.1 * learning_rate
        if epoch+1 == int(0.6*num_epochs):
            optimizer.param_groups[0]['lr'] = 0.01 * learning_rate

        if (epoch+1) % 10 == 0:
            print("Validation ...")  # should this be done every N=10 epochs
            val_running_loss, val_acc = 0.0, 0.0
            predictions = []
            targets = []

            # Compute validation loss, accuracy. Use torch.no_grad() & model.eval()
            with torch.no_grad():
                model.eval()

                hidden_state = None
                for i in range(num_dev_batches):
                    X_local_validation_minibatch, y_local_validation_minibatch = (
                        dev_X[i * batch_size: (i + 1) * batch_size, ],
                        dev_Y[i * batch_size: (i + 1) * batch_size, ],
                    )
                    X_local_minibatch = X_local_validation_minibatch.permute(1, 0, 2)
                    y_local_minibatch = torch.max(y_local_validation_minibatch, 1)[1]

                    y_pred, hidden_state = model(X_local_minibatch, hidden_state)
                    if not stateful:
                        hidden_state = None

                    val_loss = loss_function(y_pred, y_local_minibatch)

                    val_running_loss += (
                        val_loss.detach().item()
                    )  # unpacks the tensor into a scalar value
                    val_acc += model.get_accuracy(y_pred, y_local_minibatch)

                    if epoch+1 == num_epochs:
                        predictions.extend(torch.argmax(y_pred, dim=1).tolist())
                        targets.extend(y_local_minibatch.tolist())

                model.train()  # reset to train mode after iterationg through validation data
                print(
                    "Epoch:  %d | --------------- | Val Loss %.4f | Val Accuracy: %.2f | ---------------"
                    % (
                        epoch+1,
                        val_running_loss / num_dev_batches,
                        val_acc / num_dev_batches,
                    ),
                )

                val_epoch_list.append(epoch)
                val_accuracy_list.append(val_acc / num_dev_batches)
                val_loss_list.append(val_running_loss / num_dev_batches)

            if epoch+1 == num_epochs:
                confusion_mat = confusion_matrix(targets, predictions,normalize='true')
                print("Confusion Matrix:")
                print(confusion_mat)

                # Labels
                classes = ['Classical', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae']

                # Draw confusion matrix plot
                plt.figure(figsize=(8, 6))
                plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes, rotation=45)
                plt.yticks(tick_marks, classes)

                # Show data in each cube
                thresh = confusion_mat.max() / 2.
                for i in range(confusion_mat.shape[0]):
                    for j in range(confusion_mat.shape[1]):
                        plt.text(j, i, format(confusion_mat[i, j], '.3f'),
                                horizontalalignment="center",
                                color="white" if confusion_mat[i, j] > thresh else "black")

                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.tight_layout()
                if savefig:
                    plt.savefig('./result/PiecewiseDecay/confusion_mat.png')
                plt.show()

    # torch.save(model.state_dict(), './model/model.pt')

    # visualization learning rate
    plt.plot(train_epoch_list, train_lr)
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("LSTM: Learning Rate vs Epochs")
    if savefig:
        plt.savefig('./result/PiecewiseDecay/graph_lr.png')
    plt.show()

    # visualization train loss
    plt.plot(train_epoch_list, train_loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("LSTM: Train Loss vs Epochs")
    if savefig:
        plt.savefig('./result/PiecewiseDecay/graph_train_loss.png')
    plt.show()

    # visualization train accuracy
    plt.plot(train_epoch_list, train_accuracy_list, color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("LSTM: Train Accuracy vs Epochs")
    if savefig:
        plt.savefig('./result/PiecewiseDecay/graph_train_accuracy.png')
    plt.show()

    # visualization validation loss
    plt.plot(val_epoch_list, val_loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("LSTM: Loss vs Epochs")
    if savefig:
        plt.savefig('./result/PiecewiseDecay/graph_val_loss.png')
    plt.show()

    # visualization validation accuracy
    plt.plot(val_epoch_list, val_accuracy_list, color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("LSTM: Validation Accuracy vs Epochs")
    if savefig:
        plt.savefig('./result/PiecewiseDecay/graph_val_accuracy.png')
    plt.show()


if __name__ == "__main__":
    main()
