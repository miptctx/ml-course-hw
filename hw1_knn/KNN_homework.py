from typing import List
from collections import Counter

import sys
import scipy
import numpy as np
import sklearn

from numpy.linalg import norm as numpy_euclidean_norm
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


class KNNVotingClassifier:
    def __init__(self, k_neighbours: int, metric: str = 'euclidean'):
        self._allowed_metrics = {
            'euclidean': lambda x, y: numpy_euclidean_norm(x - y),
        }
        
        assert metric in self._allowed_metrics, f"Metric should be one of the {self._allowed_metrics.keys()}, got {metric}"
        
        self._metric = self._allowed_metrics[metric]
        self._k_neighbours = k_neighbours
        
        self._X, self._y = None, None
    
    def fit(self, X: np.array, y: np.array) -> None:
        '''
        When fit() method called -- model just saves the Xs and ys
        '''
        self._X = X
        self._y = y
        
    def predict(self, X: np.array) -> np.array:
        '''Non-optimized version (python loop-based)'''
        
        # Assertion check -- if model is fitted or not
        assert (self._X is not None and self._y is not None), f"Model is not fitted yet!"
        
        ys_pred: np.array = np.zeros(shape=(X.shape[0], 1)) # Predictions matrix allocation
            
        '''
        For each sample in X calculate distances to the points in self._X, using the self._metric()
        calculate distances and get K nearest points. 
        '''
        for sample_id, X_this in enumerate(X):
            distances: List = []
            
            for train_id, X_other in enumerate(self._X):
                distance = np.log(self._metric(X_this, X_other))
                distances.append({
                    'train_id': train_id,
                    'distance': distance,
                })
            sorted_distances: List = self._sort_distances(distances)
            y_pred: int = self._get_nearest_class(sorted_distances)
            ys_pred[sample_id] = y_pred

        return ys_pred
     
    @staticmethod
    def _sort_distances(distances: List, ascending=False) -> List:
        return sorted(distances, key=lambda x: x['distance'], reverse=ascending)
    
    def _get_nearest_class(self, sorted_distances: list) -> int:
        sorted_distances_top_k: List = sorted_distances[:self._k_neighbours]
        labels_top_k: List = [self._y[sample['train_id']] for sample in sorted_distances_top_k]
        predicted_label: int = self._decision_rule(labels_top_k)
        return predicted_label
    
    @staticmethod
    def _decision_rule(labels_top_k: List) -> int:
        labels_count_top_k = Counter(labels_top_k) # {label_1: label_1_num_occurences, ...}
        sorted_labels_count_top_k: List = sorted(labels_count_top_k.items(), 
                                                 key=lambda x: x[1], 
                                                 reverse=True)
        predicted_label: int = sorted_labels_count_top_k[0][0]
        return predicted_label


str_train_id = 'train_id'
str_distance = 'distance'

class KNNVotingClassifierWithWeight(KNNVotingClassifier):
    def __init__(self, k_neighbours: int, metric: str = 'euclidean', weight_samples=False):
        super().__init__(k_neighbours, metric)
        self.weight_samples = weight_samples

    def predict(self, X: np.array) -> np.array:
        '''Non-optimized version (python loop-based)'''
        
        # Assertion check -- if model is fitted or not
        assert (self._X is not None and self._y is not None), f"Model is not fitted yet!"
        
        ys_pred: np.array = np.zeros(shape=(X.shape[0], 1)) # Predictions matrix allocation

        '''
        For each sample in X calculate distances to the points in self._X, using the self._metric()
        calculate distances and get K nearest points. 
        '''
        for sample_id, X_this in enumerate(X):
            distances: List = []
            
            for train_id, X_other in enumerate(self._X):
                distance = self._metric(X_this, X_other)
                #if sample_id == 0 and train_id == 0:
                #    print(f"{X_this}, {X_other}, {distance}")

                distances.append({
                    str_train_id: train_id,
                    str_distance: distance,
                })
            sorted_distances: List = self._sort_distances(distances)
            y_pred: int = self._get_nearest_class(sorted_distances) if not self.weight_samples else self._get_nearest_class_weigth(sorted_distances)
            ys_pred[sample_id] = y_pred

        return ys_pred

    @staticmethod
    def fix_distance(value, idx, total):
        return value*(total + 0 - idx)/total

    def _get_nearest_class_weigth(self, sorted_distances: list) -> int:
        sorted_distances_top_k: List = sorted_distances[:self._k_neighbours]
        sorted_distances_top_k_weight = [
            {
                str_distance: self.fix_distance(item[str_distance], idx, len(sorted_distances_top_k)),
                str_train_id: item[str_train_id]
            }
            for idx, item in enumerate(sorted_distances_top_k)
        ]

        predicted_train_id: int = self._decision_rule_weight(sorted_distances_top_k_weight)
        return self._y[predicted_train_id]

    @staticmethod
    def _decision_rule_weight(distances: List) -> int:
        distances_weights = {}
        for item in distances:
            if item[str_train_id] in distances_weights:
                distances_weights[item[str_train_id]] += item[str_distance]
            else:
                distances_weights[item[str_train_id]] = item[str_distance]

        distances_weights_sorted = sorted(distances_weights.items(), key=lambda x: x[1], reverse=True)
        return distances_weights_sorted[0][0]



class OptimizedKNNClassifier(KNNVotingClassifierWithWeight):
    def predict(self, X: np.array) -> np.array:
        ys_pred: np.array = np.zeros(shape=(X.shape[0], 1)) # Predictions matrix allocation

        distances = []
        total = len(X)
        for sample_id, row in enumerate(X):
            sys.stdout.write(f"\r\tprocessed {sample_id} rows from {total} \t " + "{:.2f}".format(sample_id*100/total) + "%                    ")
            # calculate matrix of all distances between current tested point and all other trained points, broadcasting is made here
            distances = self._X - row

            # calc frobenius norm for each row
            distances = np.linalg.norm(distances, axis=1)

            # enumerate distances to keep their current indexes and sort
            sorted_distances = sorted(enumerate(distances), key=lambda x: x[1])[:self._k_neighbours]

            # select our nearest neighbours
            k_neighbours = np.array(list(map(lambda x: x[1], sorted_distances)))

            # select our k nearest neighbours
            neighbours_idx = np.arange(self._k_neighbours)

            # calculate weigts depending on their index,
            # i.e. calculate weighted distances;
            # a little breadcasting is made here again
            k_neighbours = k_neighbours*(self._k_neighbours - neighbours_idx)/self._k_neighbours

            # count most weighted target between our k neighbours
            counts = {}
            for i, weight in enumerate(k_neighbours):
                target = self._y[sorted_distances[i][0]]
                if target in counts:
                    counts[target] += weight
                else:
                    counts[target] = weight

            # save predicted feature value
            ys_pred[sample_id] = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]

        sys.stdout.write(f"\r\tprocessed 100%                                         \n")
        return ys_pred


class CdistOptimizedKNNClassifier(KNNVotingClassifierWithWeight):
    def predict(self, X: np.array) -> np.array:
        ys_pred: np.array = np.zeros(shape=(X.shape[0], 1)) # Predictions matrix allocation

        total = len(X)

        all_distances = np.transpose(scipy.spatial.distance.cdist(self._X, X))

        for sample_id, distances in enumerate(all_distances):
            sys.stdout.write(f"\r\tprocessed {sample_id} rows from {total} \t " + "{:.2f}".format(sample_id*100/total) + "%                    ")

            # enumerate all distances to save their current id and sort items
            sorted_distances = sorted(enumerate(distances), key=lambda x: x[1])[:self._k_neighbours]
            # print("sorted_distances\n", sorted_distances)

            # select our k nearest neighbours
            k_neighbours = np.array(list(map(lambda x: x[1], sorted_distances)))
            neighbours_idx = np.arange(self._k_neighbours)

            # calculate weigts depending on their index,
            # i.e. calculate weighted distances;
            # a little breadcasting is made here again
            k_neighbours = k_neighbours*(self._k_neighbours - neighbours_idx)/self._k_neighbours

            # count most weighted target between our k neighbours
            counts = {}
            for i, weight in enumerate(k_neighbours):
                target = self._y[sorted_distances[i][0]]
                if target in counts:
                    counts[target] += weight
                else:
                    counts[target] = weight

            # save predicted feature value
            ys_pred[sample_id] = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]

        sys.stdout.write(f"\r\tprocessed 100%                                         \n")
        return ys_pred


class CdistOptimizedKNNClassifier2(KNNVotingClassifierWithWeight):
    def predict(self, X: np.array) -> np.array:
        ys_pred: np.array = np.zeros(shape=(X.shape[0], 1)) # Predictions matrix allocation

        # calculate distances between test and train items
        all_distances = scipy.spatial.distance.cdist(self._X, X)

        # get sorted indexes of calculated distances
        k_nearest_neighbours_idx = np.argsort(all_distances, axis=0)[:self._k_neighbours,:]

        columns = k_nearest_neighbours_idx.shape[1]
        # iterate by each tested item
        for column in range(columns):
            sys.stdout.write(f"\r\tprocessed {column} columns from {columns} \t " + "{:.2f}".format(column*100/columns) + "%                    ")

            # select column of nearest neighbours
            nearest_neighbours_distances = all_distances[:,column][k_nearest_neighbours_idx[:,column]]

            # generated indexes for our nearest neighbours
            k_nearest_neighbours_arranged = np.transpose(np.arange(self._k_neighbours))

            # calculate weigts depending on their index,
            # i.e. calculate weighted distances;
            # a little breadcasting is made here
            nearest_neighbours_distances_weighted = nearest_neighbours_distances*(self._k_neighbours - k_nearest_neighbours_arranged)/self._k_neighbours

            # count most weighted targets between our k neighbours
            counts = {}
            targets = self._y[k_nearest_neighbours_idx[:,column]]
            for i in k_nearest_neighbours_arranged:
                weight = nearest_neighbours_distances_weighted[i]
                target = targets[i]
                if target in counts:
                    counts[target] += weight
                else:
                    counts[target] = weight

            # save predicted target value
            ys_pred[column] = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]

        sys.stdout.write(f"\r\tprocessed 100%                                         \n")
        return ys_pred

def train_test_split_data(X, y, train_percentage=0.7):
    num_data_points = X.shape[0]
    
    train_data_points = int(num_data_points * train_percentage)
    
    all_idx = np.arange(num_data_points)
    np.random.shuffle(all_idx)
    
    train_idx = all_idx[:train_data_points]
    test_idx = all_idx[train_data_points:]
    
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


'''
iris_dataset = datasets.load_iris()
features = iris_dataset.data
target = iris_dataset.target
np.random.seed(2)
X_train, y_train, X_test, y_test = train_test_split_data(features, target)
'''

features, target = datasets.fetch_openml("mnist_784", parser='auto', return_X_y = True, as_frame=False)
features = features[:2048]
target = target[:2048]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features, target, test_size = 0.10, random_state = 42)

y_train = np.array(y_train).astype('uint8')
y_test = np.array(y_test).astype('uint8')


k_neighbours = 5

knn_voting_clf_sklern = KNeighborsClassifier(10, metric='euclidean', algorithm='brute')
knn_voting_clf_example = KNNVotingClassifier(k_neighbours)
knn_voting_clf_custom = KNNVotingClassifierWithWeight(k_neighbours, weight_samples=True)
knn_voting_clf_weight_optimized = OptimizedKNNClassifier(k_neighbours)
knn_voting_clf_weight_cdist = CdistOptimizedKNNClassifier(k_neighbours)
knn_voting_clf_weight_cdist_2 = CdistOptimizedKNNClassifier2(k_neighbours)

def do_prediction(predictor, X_train, y_train, X_test, y_test):    
    import time
    t = time.process_time()

    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)

    elapsed_time = time.process_time() - t

    # print(f"predicted result: {y_pred.squeeze()}")

    accuracy = np.count_nonzero(y_test == y_pred.squeeze()) / y_test.shape[0]
    print(f"\taccuracy: {accuracy}")
    print(f"\telapsed time: ", "{:.6f}".format(elapsed_time))

def test_sklearn():
    print("test sklearn:")
    do_prediction(knn_voting_clf_sklern, X_train, y_train, X_test, y_test)

def test_example():
    print("test example:")
    do_prediction(knn_voting_clf_example, X_train, y_train, X_test, y_test)

def test_custom():
    print("test custom:")
    do_prediction(knn_voting_clf_custom, X_train, y_train, X_test, y_test)

def test_optimized():
    print("test optimized:")
    do_prediction(knn_voting_clf_weight_optimized, X_train, y_train, X_test, y_test)

def test_cdist():
    print("test cdist optimized:")
    do_prediction(knn_voting_clf_weight_cdist, X_train, y_train, X_test, y_test)

def test_cdist_2():
    print("test cdist v2 optimized:")
    do_prediction(knn_voting_clf_weight_cdist_2, X_train, y_train, X_test, y_test)


test_cdist_2()
test_cdist()
# test_optimized()
test_sklearn()
test_custom()
test_example()
