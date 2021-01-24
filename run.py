#!/usr/bin/env python

# import the required packages here
import os
import argparse
from data import train_data_prepare
from train import train
from test import test, test_data_prepare

<<<<<<< HEAD
def run(train_file, valid_file, test_file, output_file, batch_size, batch_size_val, epoch):
=======
def run(train_file, valid_file, test_file, output_file, batch_size, batch_size_val, n_epochs, lr, model_path):
>>>>>>> aa00c6d410e6e56d09cfce3b9a178668cf71768c
    '''The function to run your ML algorithm on given datasets, generate the output and save them into the provided file path

    Parameters
    ----------
    train_file: string
        the path to the training file
        valid_file: string
                the path to the validation file
        test_file: string
                the path to the testing file
    output_file: string
        the path to the output predictions to be saved
    '''

    ## your implementation here

    # read data from input
    train_samples, word2num, max_len_statement,  max_len_subject, max_len_speaker_pos, max_len_context = train_data_prepare(train_file)
    valid_samples = test_data_prepare(valid_file, word2num, 'valid')

    # your training algorithm
<<<<<<< HEAD
    model = train(train_samples, valid_samples, word2num, max_len_statement,  max_len_subject, max_len_speaker_pos, max_len_context, batch_size=batch_size, batch_size_val=batch_size_val, epoch = epoch)
=======
    val_acc = train(train_samples, valid_samples, word2num, max_len_statement, 
                  max_len_subject, max_len_speaker_pos, max_len_context,
                  batch_size=batch_size, batch_size_val=batch_size_val, 
                  epoch=n_epochs, lr=lr, model_path=model_path)
>>>>>>> aa00c6d410e6e56d09cfce3b9a178668cf71768c

    # your prediction code
    test(test_file, output_file, word2num,
         model_path, batch_size, lr,
         val_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Learning project')
    parser.add_argument('--data', type=str, default='data', metavar='D',
                        help="folder where data is located")
    parser.add_argument('--predictions', type=str, default='predictions', metavar='P',
                        help="folder where predictions are stored")
    parser.add_argument('--batch_size', type=int, default=20, metavar='P',
                        help="folder where predictions are stored")
<<<<<<< HEAD
    parser.add_argument('--batch_size_val', type=str, default=5, metavar='P',
                        help="folder where predictions are stored")  
    parser.add_argument('--epoch', type=int, default=5, metavar='P',
                        help="folder where predictions are stored")  
=======
    parser.add_argument('--batch_size_val', type=int, default=5, metavar='P',
                        help="folder where predictions are stored")
    parser.add_argument('--epochs', type=int, default=5, metavar='P',
                        help="number of epochs") 
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning_rate')
    parser.add_argument('--model_path', type=str, default='models', 
                      help='models')
>>>>>>> aa00c6d410e6e56d09cfce3b9a178668cf71768c
    args = parser.parse_args()

    train_path = os.path.join(args.data, 'train.tsv')
    valid_path = os.path.join(args.data, 'valid.tsv')
    test_path = os.path.join(args.data, 'test.tsv')
    predictions_path = os.path.join(args.predictions, 'predictions.txt')

    run(train_path, 
        valid_path,
        test_path,
        predictions_path,
        int(args.batch_size),
<<<<<<< HEAD
        int(args.batch_size_val), 
        int(args.epoch))
=======
        int(args.batch_size_val),
        int(args.epochs),
        args.lr,
        args.model_path)
>>>>>>> aa00c6d410e6e56d09cfce3b9a178668cf71768c
