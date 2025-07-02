import pandas as pd
import numpy as np
import ete3
from scipy import *
import jax.numpy as jnp
from jax import jit
import jax
from ete3 import Tree
import csv
from test_newick_to_vector import *

def name_internal_nodes(tree: Tree) -> Tree:
    """
    Assigns unique names to all internal nodes of a tree that do not have a name.

    Args:
        tree (Tree): ETE3 tree object.

    Returns:
        Tree: The same tree with named internal nodes.
    """
    count = 0
    for node in tree.traverse():
        if not node.is_leaf() and node.name.strip() == "":
            node.name = f"node_{count}"
            count += 1
    return tree

def get_ordered_nodes(tree: Tree) -> list:
    """
    Returns a list of all node names sorted by descending distance from the root.

    Args:
        tree (Tree): ETE3 tree object.

    Returns:
        list: List of node names sorted by descending distance to root.
    """
    node_distances = {}

    for node in tree.traverse():
        distance_to_root = node.get_distance(tree)
        node_distances[node.name] = distance_to_root

    sorted_nodes = sorted(node_distances, key=node_distances.get, reverse=True)

    return sorted_nodes

def csv_to_dates_vector(csv_file: str, node_order: list) -> np.ndarray:
    """
    Creates a date vector for the given node order, using leaf dates from a CSV file.
    Internal nodes are assigned a value of 0.

    Args:
        csv_file (str): Path to the CSV file containing columns "Leaf" and "Date".
        node_order (list): List of node names sorted by descending distance to root.

    Returns:
        np.ndarray: Date vector with values for leaves and 0 for internal nodes.
    """
    df = pd.read_csv(csv_file)
    leaf_dates = dict(zip(df["Leaf"].astype(str), df["Date"]))
    dates_vector = np.zeros(len(node_order))

    for i, node in enumerate(node_order):
        if node in leaf_dates:
            dates_vector[i] = leaf_dates[node]

    return dates_vector

def build_complete_date_vector(time_tree: Tree, node_order: list, leaf_csv: str) -> np.ndarray:
    """
    Constructs a complete date vector (for all nodes) based on a time tree and leaf date CSV.

    The root date is inferred using the first leaf's date and its distance to the root.

    Args:
        time_tree (Tree): ETE3 tree object with time distances.
        node_order (list): Ordered list of node names (farthest from root first).
        leaf_csv (str): Path to CSV file with "Leaf" and "Date" columns.

    Returns:
        np.ndarray: Vector of dates for all nodes (internal and leaves).
    """
    named_time_tree = name_internal_nodes(time_tree)
    df = pd.read_csv(leaf_csv, dtype={"Leaf": str})
    first_leaf_name = df.iloc[0]["Leaf"]
    first_leaf_date = df.iloc[0]["Date"]
    leaf_node = named_time_tree.search_nodes(name=str(first_leaf_name))[0]
    dist_leaf = leaf_node.get_distance(named_time_tree.get_tree_root())
    root_date = first_leaf_date - dist_leaf

    N = len(node_order)
    complete_vec = np.zeros(N, dtype=float)

    for i, node_name in enumerate(node_order):
        found = named_time_tree.search_nodes(name=str(node_name))
        if not found:
            complete_vec[i] = 0.0
        else:
            node_in_time_tree = found[0]
            dist_to_root = node_in_time_tree.get_distance(named_time_tree.get_tree_root())
            node_date = root_date + dist_to_root
            complete_vec[i] = node_date

    return complete_vec

if __name__ == '__main__':
    import sys

    if len(sys.argv) != 5:
        print("Usage : python3 csv_to_dates_vector.py tree.nwk leaf_dates.csv output_complete.npy output_partial.npy")
        sys.exit(1)

    tree_file = sys.argv[1]
    csv_file = sys.argv[2]
    output_complete = sys.argv[3]
    output_partial = sys.argv[4]

    tree = ete3.Tree(tree_file)
    tree = name_internal_nodes(tree)
    node_order = get_ordered_nodes(tree)

    partial_vector = csv_to_dates_vector(csv_file, node_order)
    complete_vector = build_complete_date_vector(tree, node_order, csv_file)

    np.save(output_complete, complete_vector)
    np.save(output_partial, partial_vector)

    print(f"Saved:\n- Complete vector → {output_complete}\n- Partial vector  → {output_partial}")

