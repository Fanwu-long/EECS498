"""
Implements a K-Nearest Neighbor classifier in PyTorch.
"""
import torch
from typing import Dict, List


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from knn.py!")


def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """Squared Euclidean distances with two loops."""
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function using a pair of nested loops             #
    ##########################################################################
    x_train_flat = x_train.reshape(num_train, -1)
    x_test_flat = x_test.reshape(num_test, -1)
    for i in range(num_train):
        for j in range(num_test):
            diff = x_train_flat[i] - x_test_flat[j]
            dists[i, j] = (diff * diff).sum()
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    """Squared Euclidean distances with one loop."""
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function using only a single loop over x_train.   #
    ##########################################################################
    x_train_flat = x_train.reshape(num_train, -1)
    x_test_flat = x_test.reshape(num_test, -1)
    for i in range(num_train):
        diff = x_train_flat[i] - x_test_flat
        dists[i] = (diff * diff).sum(dim=1)
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """Squared Euclidean distances with no loops."""
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function without explicit loops                   #
    ##########################################################################
    A = x_train.reshape(num_train, -1)
    B = x_test.reshape(num_test, -1)
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    aa = (A * A).sum(dim=1, keepdim=True)          # (num_train, 1)
    bb = (B * B).sum(dim=1).unsqueeze(0)           # (1, num_test)
    dists = aa - 2.0 * A.mm(B.t()) + bb
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def predict_labels(dists: torch.Tensor, y_train: torch.Tensor, k: int = 1):
    """Majority vote among k nearest neighbors; smallest label on ties."""
    num_train, num_test = dists.shape
    y_pred = torch.zeros(num_test, dtype=torch.int64)
    ##########################################################################
    # TODO: Implement this function.                                         #
    ##########################################################################
    # dists: (num_train, num_test) -> topk along train dim
    _, nn_idx = torch.topk(dists, k, dim=0, largest=False)  # (k, num_test)
    for j in range(num_test):
        labels = y_train[nn_idx[:, j]]
        counts = torch.bincount(labels)
        y_pred[j] = torch.argmax(counts)  # smallest index on ties
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return y_pred


class KnnClassifier:

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):
        ######################################################################
        # TODO: Memorize training data                                       #
        ######################################################################
        self.x_train = x_train
        self.y_train = y_train
        ######################################################################
        #                         END OF YOUR CODE                           #
        ######################################################################

    def predict(self, x_test: torch.Tensor, k: int = 1):
        y_test_pred = None
        ######################################################################
        # TODO: Predict using no-loop distances                              #
        ######################################################################
        dists = compute_distances_no_loops(self.x_train, x_test)
        y_test_pred = predict_labels(dists, self.y_train, k=k)
        ######################################################################
        #                         END OF YOUR CODE                           #
        ######################################################################
        return y_test_pred

    def check_accuracy(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        k: int = 1,
        quiet: bool = False
    ):
        y_test_pred = self.predict(x_test, k=k)
        num_samples = x_test.shape[0]
        num_correct = (y_test == y_test_pred).sum().item()
        accuracy = 100.0 * num_correct / num_samples
        msg = (
            f"Got {num_correct} / {num_samples} correct; "
            f"accuracy is {accuracy:.2f}%"
        )
        if not quiet:
            print(msg)
        return accuracy


def knn_cross_validate(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_folds: int = 5,
    k_choices: List[int] = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100],
):
    x_train_folds = []
    y_train_folds = []
    ##########################################################################
    # TODO: Split into folds                                                 #
    ##########################################################################
    x_train_folds = list(torch.chunk(x_train, num_folds, dim=0))
    y_train_folds = list(torch.chunk(y_train, num_folds, dim=0))
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################

    k_to_accuracies = {}

    ##########################################################################
    # TODO: Cross-validation                                                 #
    ##########################################################################
    for k in k_choices:
        accs = []
        for fold in range(num_folds):
            x_val = x_train_folds[fold]
            y_val = y_train_folds[fold]
            x_tr = torch.cat(
                [x_train_folds[i] for i in range(num_folds) if i != fold], dim=0
            )
            y_tr = torch.cat(
                [y_train_folds[i] for i in range(num_folds) if i != fold], dim=0
            )
            clf = KnnClassifier(x_tr, y_tr)
            accs.append(clf.check_accuracy(x_val, y_val, k=k, quiet=True))
        k_to_accuracies[k] = accs
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################

    return k_to_accuracies


def knn_get_best_k(k_to_accuracies: Dict[int, List]):
    best_k = 0
    ##########################################################################
    # TODO: Choose k with highest mean accuracy; smallest on ties            #
    ##########################################################################
    best_mean = -1.0
    for k in sorted(k_to_accuracies.keys()):
        mean_acc = sum(k_to_accuracies[k]) / len(k_to_accuracies[k])
        if mean_acc > best_mean:
            best_mean = mean_acc
            best_k = k
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return best_k
