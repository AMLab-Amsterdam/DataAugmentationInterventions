import argparse
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import torch
import matplotlib.pyplot as plt

from paper_experiments.synthetic_data.domain_gen_sem import DomainGenToyData

# Training settings
parser = argparse.ArgumentParser(description='Tpy experiment')
parser.add_argument('--samples', default=1000)
parser.add_argument('--dim', default=10)
parser.add_argument('--transform_c', default=True)
parser.add_argument('--transform_y_and_d', default=True)
parser.add_argument('--additional_noise_y_and_d', default=True)
parser.add_argument('--additional_noise_x1_and_x2', default=True)
parser.add_argument('--no_noise_d_features', default=0)
parser.add_argument('--no_noise_y_features', default=5)
args = parser.parse_args()

result_list = []
for seed in range(50):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load data
    data_loader = DomainGenToyData(args.dim,
                                transform_c=args.transform_c,
                                transform_y_and_d=args.transform_y_and_d,
                                additional_noise_y_and_d=args.additional_noise_y_and_d,
                                additional_noise_x1_and_x2=args.additional_noise_x1_and_x2)
    x_train, _, d_train = data_loader(1000, train=True)

    # Add uniform noise to causal features of d
    # Generate uniform noise [-10, 10]
    noise_x1 = torch.FloatTensor(x_train.shape[0], args.dim//2).uniform_(-10, 10).numpy()

    # Randomly select channels that get no noise
    feature_index_no_noise_x1 = np.random.choice(args.dim//2, args.no_noise_d_features, replace=False)

    # set noise to zero
    noise_x1[:, feature_index_no_noise_x1] = 0.0

    # Add to x_1
    x_train[:, :5] += noise_x1
    # x_train[:, :5] = 10

    # Add uniform noise to causal features of y
    # Generate uniform noise [-20, 20]
    noise_x2 = torch.FloatTensor(x_train.shape[0], args.dim//2).uniform_(-10, 10).numpy()

    # Randomly select channels that get no noise
    feature_index_no_noise_x2 = np.random.choice(args.dim//2, args.no_noise_y_features, replace=False)

    # set noise to zero
    noise_x2[:, feature_index_no_noise_x2] = 0.0

    # Add to x_2
    x_train[:, 5:] += noise_x2

    x_test, _, d_test = data_loader(1000, train=False)

    x_train = x_train[:, 5:]
    x_test = x_test[:, 5:]
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x_train, d_train)

    # Make predictions using the testing set
    d_pred = regr.predict(x_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(d_test, d_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(d_test, d_pred))

    # plt.figure()
    # plt.scatter(np.arange(len(y_test)), y_test)
    # plt.scatter(np.arange(len(y_test)), y_pred)
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # plt.savefig('compare_predictions.png', bbox_inches='tight')

    result_list.append(mean_squared_error(d_test, d_pred))

print(np.mean(result_list))
print(np.std(result_list)/np.sqrt(50))