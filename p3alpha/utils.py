#!/usr/bin/env python3

from p3alpha import Incremental_Similarity_Builder

import numpy as np
import scipy.sparse as sps
import time
import os


def check_matrix(X, format='csc', dtype=np.float32):
    """
    This function takes a matrix as input and transforms it into the specified format.
    The matrix in input can be either sparse or ndarray.
    If the matrix in input has already the desired format, it is returned as-is
    the dtype parameter is always applied and the default is np.float32
    :param X:
    :param format:
    :param dtype:
    :return:
    """

    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)

    elif format == 'npy':
        if sps.issparse(X):
            return X.toarray().astype(dtype)
        else:
            return np.array(X)

    elif isinstance(X, np.ndarray):
        X = sps.csr_matrix(X, dtype=dtype)
        X.eliminate_zeros()
        return check_matrix(X, format=format, dtype=dtype)
    else:
        return X.astype(dtype)


def similarityMatrixTopK(item_weights, k=100, verbose=False, use_absolute_values=False):
    """
    The function selects the TopK most similar elements, column-wise

    :param item_weights:
    :param forceSparseOutput:
    :param k:
    :param verbose:
    :param inplace: Default True, WARNING matrix will be modified
    :return:
    """

    assert (item_weights.shape[0] == item_weights.shape[1]
            ), "selectTopK: ItemWeights is not a square matrix"

    n_items = item_weights.shape[0]
    similarity_builder = Incremental_Similarity_Builder(
        n_items, initial_data_block=n_items*k, dtype=np.float32)

    start_time = time.time()

    if verbose:
        print("Generating topK matrix")

    # for each column, keep only the top-k scored items
    sparse_weights = not isinstance(item_weights, np.ndarray)

    # iterate over each column and keep only the top-k similar items
    if sparse_weights:
        item_weights = check_matrix(
            item_weights, format='csc', dtype=np.float32)

    for item_idx in range(n_items):

        if sparse_weights:
            start_position = item_weights.indptr[item_idx]
            end_position = item_weights.indptr[item_idx+1]

            column_data = item_weights.data[start_position:end_position]
            column_row_index = item_weights.indices[start_position:end_position]

        else:
            column_data = item_weights[:, item_idx]
            column_row_index = np.arange(n_items, dtype=np.int32)

        if np.any(column_data == 0):
            non_zero_data = column_data != 0
            column_data = column_data[non_zero_data]
            column_row_index = column_row_index[non_zero_data]

        # If there is less data than k, there is no need to sort
        if k < len(column_data):
            # Use argpartition because I only need to select "which" are the topK elements, I do not need their exact order
            if use_absolute_values:
                top_k_idx = np.argpartition(-np.abs(column_data),
                                            k-1, axis=0)[:k]
            else:
                top_k_idx = np.argpartition(-column_data, k-1, axis=0)[:k]

            try:
                column_row_index = column_row_index[top_k_idx]
                column_data = column_data[top_k_idx]
            except:
                pass

        similarity_builder.add_data_lists(row_list_to_add=column_row_index,
                                          col_list_to_add=np.ones(
                                              len(column_row_index), dtype=np.int) * item_idx,
                                          data_list_to_add=column_data)

    if verbose:
        print("Sparse TopK matrix generated in {:.2f} seconds".format(
            time.time() - start_time))

    return similarity_builder.get_SparseMatrix()


def seconds_to_biggest_unit(time_in_seconds):

    conversion_factor_list = [
        ("sec", 1),
        ("min", 60),
        ("hour", 60),
        ("day", 24),
        ("year", 365),
    ]

    unit_index = 0
    temp_time_value = time_in_seconds
    new_time_value = time_in_seconds
    new_time_unit = "sec"

    while temp_time_value >= 1.0 and unit_index < len(conversion_factor_list)-1:

        temp_time_value = temp_time_value / \
            conversion_factor_list[unit_index+1][1]

        if temp_time_value >= 1.0:
            unit_index += 1
            new_time_value = temp_time_value
            new_time_unit = conversion_factor_list[unit_index][0]

    else:
        return new_time_value, new_time_unit
