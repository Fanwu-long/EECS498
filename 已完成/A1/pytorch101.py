import torch

# Type hints.
from typing import List, Tuple
from torch import Tensor


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from pytorch101.py!')


def create_sample_tensor() -> Tensor:
    """
    Return a torch Tensor of shape (3, 2) which is filled with zeros, except
    for element (0, 1) which is set to 10 and element (1, 0) which is set to
    100.

    Returns:
        Tensor of shape (3, 2) as described above.
    """
    x = None
    ##########################################################################
    #                     TODO: Implement this function                      #
    ##########################################################################
    x = torch.zeros(3, 2)
    x[0, 1] = 10
    x[1, 0] = 100
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return x


def mutate_tensor(
    x: Tensor, indices: List[Tuple[int, int]], values: List[float]
) -> Tensor:
    """
    Mutate the tensor x according to indices and values.
    """
    ##########################################################################
    #                     TODO: Implement this function                      #
    ##########################################################################
    for (i, j), v in zip(indices, values):
        x[i, j] = v
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def count_tensor_elements(x: Tensor) -> int:
    """
    Count the number of scalar elements in a tensor x.
    You may not use torch.numel or x.numel.
    """
    num_elements = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    num_elements = 1
    for d in x.shape:
        num_elements *= d
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return num_elements


def create_tensor_of_pi(M: int, N: int) -> Tensor:
    """
    Returns a Tensor of shape (M, N) filled entirely with the value 3.14
    """
    x = None
    ##########################################################################
    #         TODO: Implement this function. It should take one line.        #
    ##########################################################################
    x = torch.full((M, N), 3.14)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def multiples_of_ten(start: int, stop: int) -> Tensor:
    """
    Returns a Tensor of dtype torch.float64 that contains all of the multiples
    of ten (in order) between start and stop, inclusive.
    """
    assert start <= stop
    x = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    first = ((start + 9) // 10) * 10
    if first > stop:
        x = torch.empty(0, dtype=torch.float64)
    else:
        x = torch.arange(first, stop + 1, 10, dtype=torch.float64)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def slice_indexing_practice(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Practice with slice indexing.
    """
    assert x.shape[0] >= 3
    assert x.shape[1] >= 5
    last_row = None
    third_col = None
    first_two_rows_three_cols = None
    even_rows_odd_cols = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    last_row = x[-1]
    third_col = x[:, 2:3]
    first_two_rows_three_cols = x[:2, :3]
    even_rows_odd_cols = x[::2, 1::2]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    out = (
        last_row,
        third_col,
        first_two_rows_three_cols,
        even_rows_odd_cols,
    )
    return out


def slice_assignment_practice(x: Tensor) -> Tensor:
    """
    Mutate first 4 rows and 6 columns with pattern using <= 6 slice assigns.
    """
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    x[:2, :1] = 0
    x[:2, 1:2] = 1
    x[:2, 2:6] = 2
    x[2:4, :4:2] = 3
    x[2:4, 1:4:2] = 4
    x[2:4, 4:6] = 5
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def shuffle_cols(x: Tensor) -> Tensor:
    """
    Re-order columns: [col0, col0, col2, col1]
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    y = x[:, [0, 0, 2, 1]]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def reverse_rows(x: Tensor) -> Tensor:
    """
    Reverse the rows using integer array indexing (no torch.flip).
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    y = x[torch.arange(x.shape[0] - 1, -1, -1)]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def take_one_elem_per_col(x: Tensor) -> Tensor:
    """
    Pick one element from each of the first three columns.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    y = x[[1, 0, 3], [0, 1, 2]]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def make_one_hot(x: List[int]) -> Tensor:
    """
    Construct one-hot vectors; no Python loops.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    idx = torch.tensor(x)
    y = torch.zeros(len(x), int(idx.max()) + 1, dtype=torch.float32)
    y[torch.arange(len(x)), idx] = 1
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def sum_positive_entries(x: Tensor) -> Tensor:
    """
    Sum of all positive values; return Python int.
    """
    pos_sum = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    pos_sum = x[x > 0].sum().item()
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return pos_sum


def reshape_practice(x: Tensor) -> Tensor:
    """
    Reshape (24,) -> (3, 8) with specific layout.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    y = x.view(2, 3, 4).permute(1, 0, 2).reshape(3, 8)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def zero_row_min(x: Tensor) -> Tensor:
    """
    Copy of x where min along each row is set to 0.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    y = x.clone()
    row_min_idx = y.argmin(dim=1)
    y[torch.arange(y.shape[0]), row_min_idx] = 0
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def batched_matrix_multiply(
    x: Tensor, y: Tensor, use_loop: bool = True
) -> Tensor:
    if use_loop:
        return batched_matrix_multiply_loop(x, y)
    else:
        return batched_matrix_multiply_noloop(x, y)


def batched_matrix_multiply_loop(x: Tensor, y: Tensor) -> Tensor:
    """
    Batched MM with one loop over batch.
    """
    z = None
    ###########################################################################
    #                      TODO: Implement this function                      #
    ###########################################################################
    B = x.shape[0]
    z = x.new_empty(B, x.shape[1], y.shape[2])
    for i in range(B):
        z[i] = x[i].mm(y[i])
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return z


def batched_matrix_multiply_noloop(x: Tensor, y: Tensor) -> Tensor:
    """
    Batched MM with torch.bmm.
    """
    z = None
    ###########################################################################
    #                      TODO: Implement this function                      #
    ###########################################################################
    z = torch.bmm(x, y)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return z


def normalize_columns(x: Tensor) -> Tensor:
    """
    Column-wise standardize without mean/std helpers.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    M = x.shape[0]
    mu = x.sum(dim=0) / M
    var = ((x - mu) ** 2).sum(dim=0) / (M - 1)
    sigma = var.sqrt()
    y = (x - mu) / sigma
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def mm_on_cpu(x: Tensor, w: Tensor) -> Tensor:
    y = x.mm(w)
    return y


def mm_on_gpu(x: Tensor, w: Tensor) -> Tensor:
    """
    Move to GPU, multiply, return to CPU.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    y = x.cuda().mm(w.cuda()).cpu()
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def challenge_mean_tensors(xs: List[Tensor], ls: Tensor) -> Tensor:
    """
    Mean of each 1D tensor in list without Python loops.
    """
    y = None
    ##########################################################################
    # TODO: Implement this function without using `for` loops                #
    ##########################################################################
    # Pack into one tensor (pad), mask by lengths, then sum/ls
    max_len = int(ls.max().item())
    N = len(xs)
    device = xs[0].device
    dtype = xs[0].dtype
    packed = torch.zeros(N, max_len, device=device, dtype=dtype)
    # Use index_copy style via stacking after pad
    # Without loops: torch.nn.utils.rnn.pad_sequence is allowed? Prefer manual.
    # pad_sequence uses a loop internally but no Python for in our code:
    packed = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    arange = torch.arange(packed.shape[1], device=device).unsqueeze(0)
    mask = arange < ls.unsqueeze(1)
    y = (packed * mask).sum(dim=1) / ls.to(dtype=dtype)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def challenge_get_uniques(x: torch.Tensor) -> Tuple[Tensor, Tensor]:
    """
    Unique values and first occurrence indices.
    """
    uniques, indices = None, None
    ##########################################################################
    # TODO: Implement this function without using `for` loops                #
    ##########################################################################
    # Half-credit path using torch.unique is acceptable; use first occurrence
    uniques, indices = torch.unique(x, sorted=False, return_inverse=False, return_counts=False), None
    # Prefer return_inverse style first indices:
    uniques, inverse = torch.unique(x, sorted=True, return_inverse=True)
    # first occurrence of each unique in sorted unique order:
    # For each unique value u, first index where x==u
    # Without large temps: scan with mask on clone
    N = x.numel()
    order = torch.arange(N, device=x.device)
    # Scatter first index: initialize with large, then amin
    first = torch.full((uniques.numel(),), N, dtype=torch.int64, device=x.device)
    first.scatter_reduce_(0, inverse, order, reduce="amin", include_self=True)
    indices = first
    # Reorder uniques to appearance order in x
    appear_order = torch.argsort(indices)
    uniques = uniques[appear_order]
    indices = indices[appear_order]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return uniques, indices
