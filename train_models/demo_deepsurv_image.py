#!/usr/bin/env python
"""
At the end of the demo, we save the embeddings and other auxiliary output
to use with our anchor direction visualization framework.

Note that this code is written so that it can be used even with an encoder
neural net that does not project to a hypersphere, although we will be doing
this hyperspherical projection.
"""
import ast
import configparser
import csv
import gc
import h5py
import hashlib
import numpy as np
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from lifelines.utils import concordance_index
from pycox.models.loss import CoxPHLoss

from datasets import load_dataset
from common import make_relu_mlp, Hypersphere, apply_network, breslow_estimate


estimator_name = 'deepsurv'

if not (len(sys.argv) == 2 and os.path.isfile(sys.argv[1])):
    print('Usage: python "%s" [config file]' % sys.argv[0])
    sys.exit(1)

config = configparser.ConfigParser()
config.read(sys.argv[1])

n_experiment_repeats = int(config['DEFAULT']['n_experiment_repeats'])
val_ratio = float(config['DEFAULT']['simple_data_splitting_val_ratio'])
datasets = ast.literal_eval(config['DEFAULT']['datasets'])
output_dir = config['DEFAULT']['output_dir']
method_header = 'method: %s' % estimator_name
method_random_seed = int(config[method_header]['random_seed'])
patience = int(config[method_header]['early_stopping_patience'])

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_n_epochs = int(config[method_header]['max_n_epochs'])
hypersphere = int(config[method_header]['unit_hypersphere_projection'])
hyperparams = \
    [(hypersphere, n_nodes, batch_size, max_n_epochs, lr)
     for n_nodes
     in ast.literal_eval(config[method_header]['n_nodes'])
     for batch_size
     in ast.literal_eval(config[method_header]['batch_size'])
     for lr
     in ast.literal_eval(config[method_header]['learning_rate'])]

hyperparam_hash = hashlib.sha256()
hyperparam_hash.update(str(hyperparams).encode('utf-8'))
hyperparam_hash = hyperparam_hash.hexdigest()

validation_string = 'vr%f' % val_ratio

output_test_table_filename \
    = os.path.join(output_dir,
                   '%s_experiments%d_%s_test_metrics_%s.csv'
                   % (estimator_name,
                      n_experiment_repeats,
                      validation_string,
                      hyperparam_hash))
output_test_table_file = open(output_test_table_filename, 'w')
test_csv_writer = csv.writer(output_test_table_file)
test_csv_writer.writerow(['dataset',
                          'experiment_idx',
                          'method',
                          'loss'])



for experiment_idx in range(n_experiment_repeats):
    for dataset in datasets:
        X_train, y_train, X_test, y_test, feature_names, \
            compute_features_and_transformer, transform_features, \
            fixed_train_test_split \
            = load_dataset(dataset, experiment_idx,
                           fix_test_shuffle_train=True)

        y_train_digits = y_train[:, 2]
        y_test_digits = y_test[:, 2]
        y_train = y_train[:, :2]
        y_test = y_test[:, :2]

        # load_dataset already shuffles; no need to reshuffle
        proper_train_idx, val_idx = train_test_split(range(len(X_train)),
                                                     test_size=val_ratio,
                                                     shuffle=False)
        X_proper_train = X_train[proper_train_idx]
        y_proper_train = y_train[proper_train_idx].astype('float32')
        X_val = X_train[val_idx]
        y_val = y_train[val_idx].astype('float32')

        X_proper_train_std, transformer = \
            compute_features_and_transformer(X_proper_train)
        X_val_std = transform_features(X_val, transformer)
        X_proper_train_std = X_proper_train_std.astype('float32')
        X_val_std = X_val_std.astype('float32')

        print('[Dataset: %s (size=%d, raw dim=%d, dim=%d), experiment: %d]'
              % (dataset, len(X_train) + len(X_test), X_train.shape[1],
                 X_proper_train_std.shape[1], experiment_idx))
        print()

        output_train_metrics_filename \
            = os.path.join(output_dir, 'train',
                           '%s_%s_exp%d_%s_train_metrics_%s.txt'
                           % (estimator_name, dataset, experiment_idx,
                              validation_string, hyperparam_hash))
        output_best_hyperparam_filename \
            = os.path.join(output_dir, 'train',
                           '%s_%s_exp%d_%s_best_hyperparams_%s.pkl'
                           % (estimator_name, dataset, experiment_idx,
                              validation_string, hyperparam_hash))
        if not os.path.isfile(output_train_metrics_filename) or \
                not os.path.isfile(output_best_hyperparam_filename):
            print('Training...', flush=True)
            train_metrics_file = open(output_train_metrics_filename, 'w')
            best_hyperparams = {}

            min_loss = np.inf
            arg_min = None
            best_model_filename = None

            for hyperparam_idx, hyperparam in enumerate(hyperparams):
                hypersphere, n_nodes, batch_size, max_n_epochs, lr \
                    = hyperparam

                # seed different hyperparameters differently to prevent weird
                # behavior where a bad initial seed makes a specific model
                # always look terrible
                hyperparam_random_seed = method_random_seed + hyperparam_idx

                tic = time.time()
                torch.manual_seed(hyperparam_random_seed)
                np.random.seed(hyperparam_random_seed)
                random.seed(hyperparam_random_seed)

                layers = [nn.Conv2d(1, 32, 3),
                          nn.ReLU(),
                          nn.MaxPool2d(2),
                          nn.Conv2d(32, 16, 3),
                          nn.ReLU(),
                          nn.MaxPool2d(2),
                          nn.Flatten(),
                          nn.Linear(400, n_nodes)]  # 400 is specific to MNIST
                if hypersphere > 0:
                    layers.append(Hypersphere())
                else:
                    layers.append(nn.ReLU())
                encoder_net = nn.Sequential(*layers)
                net = nn.Sequential(encoder_net,
                                    nn.Linear(n_nodes, 1, bias=False)).to(device)
                optimizer = torch.optim.Adam(net.parameters(), lr=lr)

                model_filename = \
                    os.path.join(
                        output_dir, 'models',
                        '%s_%s_exp%d_h%d_'
                        % (estimator_name, dataset, experiment_idx,
                           hypersphere)
                        +
                        'nno%d_bs%d_mnep%d_lr%f.pt'
                        % (n_nodes, batch_size, max_n_epochs, lr))
                time_elapsed_filename = \
                    model_filename[:-3] + '_time.txt'
                epoch_time_elapsed_filename = \
                    model_filename[:-3] + '_epoch_times.txt'
                epoch_times = []

                min_loss_current_hyperparam = np.inf
                train_loader = \
                    torch.utils.data.DataLoader(
                        list(zip(torch.tensor(X_proper_train_std,
                                              dtype=torch.float32),
                                 torch.tensor(y_proper_train,
                                              dtype=torch.float32))),
                        batch_size=batch_size,
                        shuffle=True)

                cox_loss = CoxPHLoss()
                for epoch_idx in range(max_n_epochs):
                    tic_ = time.time()

                    for batch_inputs, batch_targets in train_loader:
                        batch_inputs = batch_inputs.to(device)
                        batch_targets = batch_targets.to(device)

                        batch_outputs = net(batch_inputs)
                        batch_loss = cox_loss(batch_outputs,
                                              batch_targets[:, 0],
                                              batch_targets[:, 1])

                        optimizer.zero_grad()
                        batch_loss.backward()
                        optimizer.step()

                    epoch_train_time = time.time() - tic_

                    tic_ = time.time()
                    y_val_pred = apply_network(net, X_val_std, batch_size, device)
                    val_loss = \
                        -concordance_index(y_val[:, 0], -y_val_pred, y_val[:, 1])
                    epoch_val_time = time.time() - tic_

                    epoch_times.append([epoch_train_time, epoch_val_time])

                    new_hyperparam = \
                        (hypersphere, n_nodes, batch_size,
                         epoch_idx + 1, lr, hyperparam_random_seed)
                    print(new_hyperparam,
                          '--',
                          'val c-index %f' % -val_loss,
                          '--',
                          'train time %f sec(s)'
                          % epoch_train_time,
                          '--',
                          'val time %f sec(s)' % epoch_val_time,
                          flush=True)
                    print(new_hyperparam, ':', val_loss, flush=True,
                          file=train_metrics_file)

                    if val_loss < min_loss_current_hyperparam:
                        min_loss_current_hyperparam = val_loss
                        wait_idx = 0
                        torch.save(net.state_dict(), model_filename)

                        if val_loss < min_loss:
                            min_loss = val_loss
                            arg_min = new_hyperparam
                            best_model_filename = model_filename
                    else:
                        wait_idx += 1
                        if patience > 0 and wait_idx >= patience:
                            break

                np.savetxt(epoch_time_elapsed_filename, np.array(epoch_times))

                elapsed = time.time() - tic
                print('Time elapsed: %f second(s)' % elapsed, flush=True)
                np.savetxt(time_elapsed_filename,
                           np.array(elapsed).reshape(1, -1))

            train_metrics_file.close()

            best_hyperparams['loss'] = (arg_min, min_loss)
            with open(output_best_hyperparam_filename, 'wb') as pickle_file:
                pickle.dump(best_hyperparams, pickle_file,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('Loading previous validation results...', flush=True)
            with open(output_best_hyperparam_filename, 'rb') as pickle_file:
                best_hyperparams = pickle.load(pickle_file)
            arg_min, min_loss = best_hyperparams['loss']

        print('Best hyperparameters for minimizing loss:',
              arg_min, '-- achieves val c-index %f' % -min_loss, flush=True)

        print()

        # ---------------------------------------------------------------------
        # Load best model
        #

        hypersphere, n_nodes, batch_size, n_epochs, lr, seed \
            = arg_min

        tic = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        layers = [nn.Conv2d(1, 32, 3),
                  nn.ReLU(),
                  nn.MaxPool2d(2),
                  nn.Conv2d(32, 16, 3),
                  nn.ReLU(),
                  nn.MaxPool2d(2),
                  nn.Flatten(),
                  nn.Linear(400, n_nodes)]  # 400 is specific to MNIST
        if hypersphere > 0:
            layers.append(Hypersphere())
        else:
            layers.append(nn.ReLU())
        encoder_net = nn.Sequential(*layers)
        net = nn.Sequential(encoder_net,
                            nn.Linear(n_nodes, 1, bias=False)).to(device)

        model_filename = \
            os.path.join(
                output_dir, 'models',
                '%s_%s_exp%d_h%d_'
                % (estimator_name, dataset, experiment_idx, hypersphere)
                +
                'nno%d_bs%d_mnep%d_lr%f.pt'
                % (n_nodes, batch_size, max_n_epochs, lr))
        net.load_state_dict(torch.load(model_filename))

        time_elapsed_filename = model_filename[:-3] + '_time.txt'
        elapsed = float(np.loadtxt(time_elapsed_filename))
        print('Time elapsed (from previous fitting): %f second(s)'
              % elapsed, flush=True)

        print()

        y_proper_train_pred = \
            apply_network(net, X_proper_train_std, batch_size, device)
        sorted_unique_times, baseline_cumulative_hazards = \
            breslow_estimate(y_proper_train, y_proper_train_pred)

        # ---------------------------------------------------------------------
        # Test set prediction
        #

        print('Testing...', flush=True)
        X_test_std = transform_features(X_test, transformer)
        X_test_std = X_test_std.astype('float32')
        y_test = y_test.astype('float32')

        y_test_pred = apply_network(net, X_test_std, batch_size, device)
        loss = -concordance_index(y_test[:, 0], -y_test_pred, y_test[:, 1])
        print('Hyperparameter', arg_min, 'achieves test c-index %f' % -loss,
              flush=True)

        test_csv_writer.writerow(
            [dataset, experiment_idx, estimator_name, loss])

        y_test_surv = np.exp(-np.outer(np.exp(y_test_pred),
                                       baseline_cumulative_hazards))

        # ---------------------------------------------------------------------
        # Save embeddings for explanation framework
        #

        print('Saving embeddings...')
        test_embeddings = \
            apply_network(encoder_net, X_test_std, batch_size, device)
        cached_embeddings_dir = os.path.join(
            output_dir,
            'cached_embeddings/%s_exp%d' % (dataset, experiment_idx))
        os.makedirs(cached_embeddings_dir, exist_ok=True)
        np.savetxt(os.path.join(cached_embeddings_dir, 'test_embeddings.txt'),
                   test_embeddings)
        np.save(os.path.join(cached_embeddings_dir, 'feature_names.npy'),
                np.array(feature_names))
        np.save(os.path.join(cached_embeddings_dir, 'test_X.npy'), X_test)
        np.savetxt(os.path.join(cached_embeddings_dir, 'test_y.txt'), y_test)
        np.savetxt(os.path.join(cached_embeddings_dir,
                                'train_unique_event_times.txt'),
                   sorted_unique_times)
        np.savetxt(os.path.join(cached_embeddings_dir, 'test_pred_surv.txt'),
                   y_test_surv)
        np.savetxt(os.path.join(cached_embeddings_dir, 'test_digits.txt'),
                   y_test_digits)

        print()
        print()
