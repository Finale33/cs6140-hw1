import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge


class HomeWork1:
    def __init__(self):
        self.pre_process = preprocessing.StandardScaler()
        pd.set_option('display.max_columns', None)
        self.dataset_root_location = 'dataset/'

    # function to get parameters from command line
    def get_parameters(self):
        while True:
            print('please type in the name for the dataset')
            file_name = input()
            file_location = self.dataset_root_location + file_name
            can_break = True
            try:
                pd.read_csv(file_location)
            except FileNotFoundError:
                print('file does not exist!')
                can_break = False
            if can_break:
                break
        while True:
            print('please type in the percentage for the test set, the number should be from 0 to 100')
            percentage_test = input()
            if not percentage_test.isnumeric():
                print('please provide correct number')
            else:
                percentage_test = int(percentage_test)
                if percentage_test >= 100 or percentage_test <= 0:
                    print('please provide correct number')
                else:
                    break
        return file_name, percentage_test

    # function to divide the dataset to a test dataset and training dataset.
    def split_dataset(self, file_name, percentage_test):
        file_location = self.dataset_root_location + file_name
        data = pd.read_csv(file_location)
        print(data.head())
        percentage_test = percentage_test * 0.01
        percentage_training = 1 - percentage_test
        self.test_set = data.sample(frac=percentage_test)
        self.training_set = data.drop(self.test_set.index)
        self.test_set.reset_index(drop=True, inplace=True)
        self.training_set.reset_index(drop=True, inplace=True)
        print('the percentage of test set is: ' + str(percentage_test) + ', and the number of sample is: ' + str(
            len(self.test_set.index)))
        print('the percentage of test set is: ' + str(percentage_training) + ', and the number of sample is: ' + str(
            len(self.training_set.index)))

        # step to save the dataset to dataset folder and print out the name
        test_set_name = 'test_set.csv'
        training_set_name = 'training_set.csv'
        self.test_set.to_csv(self.dataset_root_location + test_set_name)
        self.training_set.to_csv(self.dataset_root_location + training_set_name)
        print(
            'the name for the test set is ' + test_set_name + ', and the name for the training set is ' + training_set_name)
        print('they are both saved under dataset folder')

    # function to plot the data in pairs
    def plot_pairs(self):
        y = self.training_set['Price']
        for i in range(5):
            plt.scatter(self.training_set.iloc[:, i], y)
            plt.show()

    # function to perform linear regression on each pair
    def simple_linear_regression(self):
        lr = LinearRegression()
        y = self.training_set['Price']
        for i in range(5):
            column_name = self.training_set.columns[i]
            lr.fit(self.training_set[[column_name]], y)
            coefficient = lr.coef_
            intercept = lr.intercept_
            r_sq = lr.score(self.training_set[[column_name]], y)
            print(f"coefficient for {column_name}: {coefficient}")
            print(f"intercept for {column_name}: {intercept}")
            print(f"r_sq for {column_name}: {r_sq}")
            plt.scatter(self.training_set[column_name], y)
            plt.plot(self.training_set[column_name], coefficient * self.training_set[column_name] + intercept, 'ro')
            plt.show()

    # multiple linear regression on all independent variables
    def multiple_linear_regression(self, X, y):
        lr = LinearRegression()
        X = pd.DataFrame(self.pre_process.fit_transform(X))
        lr.fit(X, y)
        coefficient = lr.coef_
        intercept = lr.intercept_
        r_sq = lr.score(X, y)
        print('coefficient for all variables: {}'.format(coefficient))
        print('intercept for all variables: {}'.format(intercept))
        print(f"r_sq for all variables: {r_sq}")

    # ridge linear regression on all independent variables
    def ridge_linear_regression(self):
        ridge_lr = Ridge(alpha=0.5)
        y = self.training_set['Price']
        X = self.training_set[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                               'Avg. Area Number of Bedrooms', 'Area Population']]
        X = pd.DataFrame(self.pre_process.fit_transform(X))
        ridge_lr.fit(X, y)
        coefficient = ridge_lr.coef_
        intercept = ridge_lr.intercept_
        r_sq = ridge_lr.score(X, y)
        print('coefficient for all variables in ridge regression: {}'.format(coefficient))
        print('intercept for all variables in ridge regression: {}'.format(intercept))
        print(f"r_sq for all variables in ridge regression: {r_sq}")

    # function to compute up to the 3rd degree values of X data starting from degree of 0.
    def compute_3rd_degree_X(self, X_data):
        res = []
        for X in X_data:
            cur = []
            for i in range(4):
                cur.append(X ** i)
            res.append(cur)
        return res

    # function to plot 3rd degree polynomial model
    def plot_3rd_degree_polynomial(self, coefficients, X_data, y_data):
        y_pred = []
        X_range = np.linspace(20000, 100000, 3000)
        for X in X_range:
            y = coefficients[0] * X ** 0 + coefficients[1] * X ** 1 + coefficients[2] * X ** 2 + coefficients[
                3] * X ** 3
            y_pred.append(y)
        plt.scatter(X_data, y_data)
        plt.plot(X_range, y_pred, 'ro')
        plt.show()

    # Execute a linear regression with a polynomial model
    def polynomial_linear_regression(self):
        X = self.training_set['Avg. Area Income']
        y = self.training_set['Price']
        linear_squared_cubic_X = self.compute_3rd_degree_X(X)
        coefficients = np.linalg.lstsq(np.array(linear_squared_cubic_X), np.array(y), rcond=None)
        print(coefficients[0])
        self.plot_3rd_degree_polynomial(coefficients[0], X, y)

    # function to compute the principal components of the given data
    def pca(self, data, normalize=True):
        N = len(data)
        A = np.matrix(data)[:, :-1]
        m = np.mean(A, axis=0)
        print("means are ")
        print(m)
        D = np.subtract(A, m)

        if normalize:
            std = np.std(D, axis=0)
        else:
            std = np.matrix([1. for _ in range(A.shape[1])])
        print("std are: ")
        print(std)

        # divide each column by its std
        D = np.divide(D, std)
        print("D is: ")
        print(D)
        # assign to U, S, V the result of running np.svd on D, with full_matrices=False
        U, S, V = np.linalg.svd(D, full_matrices=False)

        # compute the eigenvector
        eigenvalues = S ** 2 / (N - 1)
        print("eigenvalues are: ")
        print(eigenvalues)
        print("eigenvectors are ")
        print(V)

        # project the data onto the eigenvector. Treat V as a transformation matrix
        # and right-multiply it by D transpose. The eigenvectors of A are the rows of
        # V. The eigenvectors match the order of the eigenvalues.
        projected_data = D @ V.T
        print("projected data: ")
        print(projected_data)

        # return the means, standard deviations, eigenvalues, eigenvectors, and projected data
        return m, std, eigenvalues, V, projected_data

    # main function
    def main(self):
        file_name, percentage_test = self.get_parameters()
        self.split_dataset(file_name, percentage_test)
        self.plot_pairs()
        self.simple_linear_regression()
        y = self.training_set['Price']
        X = self.training_set[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                               'Avg. Area Number of Bedrooms', 'Area Population']]
        self.multiple_linear_regression(X, y)
        self.ridge_linear_regression()
        self.polynomial_linear_regression()
        # test PCA function
        file_location = self.dataset_root_location + 'pcatestdata.csv'
        pca_data = pd.read_csv(file_location)
        # test non_whitened data
        self.pca(pca_data, False)
        # test whitened data
        self.pca(pca_data, True)
        # test pca function on USA_Housing dataset
        # test for non_whitened data
        whitened_m, whitened_std, whitened_eigenvalues, whitened_eigenvectors, whitened_projected_data = self.pca(self.training_set, False)
        # test for whitened data
        self.pca(self.training_set, True)
        y = self.training_set['Price']
        whitened_projected_data = np.asarray(whitened_projected_data).ravel().reshape((3000, 5))
        self.training_set = whitened_projected_data
        self.multiple_linear_regression(self.training_set, y)


if __name__ == "__main__":
    HomeWork1().main()
