from test_newick_to_vector import *
from simulate_time_trees import *
from time_to_genetic_distance import *
from setup_data import *
import pandas as pd
import numpy as np
import tensorflow as tf
from ete3 import Tree


# ===============================
# Padding utilities
# ===============================

def compute_max_lengths_for_all(jax_vectors, complete_date_vectors, partial_date_vectors):
    """
    Computes the maximum lengths among all tree vectors and date vectors.
    """
    max_len_jax = max(len(v) for v in jax_vectors)
    max_len_complete = max(len(d) for d in complete_date_vectors)
    max_len_partial = max(len(d) for d in partial_date_vectors)
    max_len_dates = max(max_len_complete, max_len_partial)
    return max_len_jax, max_len_dates

def pad_jax_and_dates(jax_vec, date_vec, max_len_jax, max_len_dates):
    """
    Pads a JAX tree vector and date vector with zeros to match max lengths.

    Parameters:
    - jax_vec: jnp.array or np.array
    - date_vec: np.array
    - max_len_jax: int
    - max_len_dates: int

    Returns:
    - padded_jax: np.array
    - padded_date: np.array
    """
    jax_np = np.array(jax_vec)
    date_np = np.array(date_vec)

    jax_padded = np.zeros(max_len_jax, dtype=jax_np.dtype)
    date_padded = np.zeros(max_len_dates, dtype=date_np.dtype)

    jax_padded[:len(jax_np)] = jax_np
    date_padded[:len(date_np)] = date_np

    return jax_padded, date_padded

def pad_dataset(jax_vectors, date_vectors):
    """
    Pads all jax vectors and date vectors in the dataset.
    Returns two 2D numpy arrays of shape:
        - jax_array: (n_samples, max_len_jax)
        - date_array: (n_samples, max_len_dates)
    """
    max_len_jax, max_len_dates = compute_max_lengths(jax_vectors, date_vectors)

    jax_array = []
    date_array = []
    for jax_vec, date_vec in zip(jax_vectors, date_vectors):
        jax_pad, date_pad = pad_jax_and_dates(jax_vec, date_vec, max_len_jax, max_len_dates)
        jax_array.append(jax_pad)
        date_array.append(date_pad)

    jax_array = np.stack(jax_array, axis=0)
    date_array = np.stack(date_array, axis=0)

    return jax_array, date_array


# ===============================
# Vector building utilities
# ===============================

def name_internal_nodes(tree):
    """
    Assigns unique names to internal unnamed nodes of a tree.
    """
    count = 0
    for node in tree.traverse():
        if not node.is_leaf() and node.name.strip() == "":
            node.name = f"node_{count}"
            count += 1
    return tree

def get_ordered_nodes(tree):
    """
    Returns a list of node names ordered by decreasing distance from the root.
    """
    node_distances = {}
    for node in tree.traverse():
        node_distances[node.name] = node.get_distance(tree)
    sorted_nodes = sorted(node_distances, key=node_distances.get, reverse=True)
    print(f"Number of nodes: {len(sorted_nodes)}")
    return sorted_nodes

def build_complete_date_vector(time_tree_file, leaf_csv_file, node_order):
    """
    Builds the full date vector by assigning estimated dates to each node
    based on their distance from the root and the first leaf's known date.

    Parameters:
    - time_tree_file: path to the tree file (.nwk) or ete3.Tree
    - leaf_csv_file: path to the leaf CSV with true leaf dates
    - node_order: list of ordered node names

    Returns:
    - np.array of dates (float)
    """
    if isinstance(time_tree_file, str):
        tree = Tree(time_tree_file, format=1)
    else:
        tree = time_tree_file

    tree = name_internal_nodes(tree)
    df = pd.read_csv(leaf_csv_file, dtype={"Leaf": str})
    first_leaf = df.iloc[0]
    root = tree.get_tree_root()
    leaf_node = tree.search_nodes(name=first_leaf["Leaf"])[0]
    dist = leaf_node.get_distance(root)
    root_date = first_leaf["Date"] - dist

    vec = []
    for node_name in node_order:
        found = tree.search_nodes(name=node_name)
        if found:
            node = found[0]
            d = node.get_distance(root)
            vec.append(root_date + d)
        else:
            vec.append(0.0)
    print(f"True date vector: {vec}")
    return np.array(vec)


# ===============================
# Dataset generation
# ===============================

def pad_dataset_extended(jax_vectors, complete_date_vectors, partial_date_vectors):
    """
    Pads JAX tree vectors, complete date vectors, and partial date vectors
    into 2D arrays suitable for model training.

    Returns:
    - X: shape (n_samples, max_len_jax)
    - y_complete: shape (n_samples, max_len_dates)
    - y_partial: shape (n_samples, max_len_dates)
    """
    max_len_jax, max_len_dates = compute_max_lengths_for_all(jax_vectors, complete_date_vectors, partial_date_vectors)
    print(f"Max JAX vector length: {max_len_jax}")
    print(f"Max date vector length: {max_len_dates}")

    jax_array = []
    complete_array = []
    partial_array = []

    for jax_vec, comp_vec, part_vec in zip(jax_vectors, complete_date_vectors, partial_date_vectors):
        jax_pad, comp_pad = pad_jax_and_dates(jax_vec, comp_vec, max_len_jax, max_len_dates)
        _, part_pad = pad_jax_and_dates(jax_vec, part_vec, max_len_jax, max_len_dates)
        jax_array.append(jax_pad)
        complete_array.append(comp_pad)
        partial_array.append(part_pad)

    return (
        np.stack(jax_array, axis=0),
        np.stack(complete_array, axis=0),
        np.stack(partial_array, axis=0)
    )


def pad_and_concatenate(jax_vectors, complete_date_vectors):
    """
    Pads and concatenates each JAX tree vector with its corresponding date vector.

    Returns:
    - dataset: shape (n_samples, max_len_jax + max_len_dates)
    """
    max_len_jax, max_len_dates = compute_max_lengths(jax_vectors, complete_date_vectors)

    dataset = []
    for jax_vec, date_vec in zip(jax_vectors, complete_date_vectors):
        jax_pad, date_pad = pad_jax_and_dates(jax_vec, date_vec, max_len_jax, max_len_dates)
        concat_vec = np.concatenate([jax_pad, date_pad])
        dataset.append(concat_vec)

    return np.stack(dataset, axis=0)


def create_tf_dataset(dataset_np, batch_size=32, shuffle=True):
    """
    Creates a TensorFlow dataset from a NumPy array, ready for training.

    Parameters:
    - dataset_np: NumPy array of shape (n_samples, n_features)
    - batch_size: size of each batch
    - shuffle: whether to shuffle the dataset

    Returns:
    - tf.data.Dataset
    """
    ds = tf.convert_to_tensor(dataset_np, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices(ds)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataset_np))
    ds = ds.batch(batch_size)
    return ds

