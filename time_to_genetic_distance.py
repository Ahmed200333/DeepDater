#!/usr/bin/env python3

import sys
import numpy as np
from ete3 import Tree

def time_tree_to_genetic_tree(time_tree_nwk: str, output_nwk: str = "genetic_tree.nwk"):
    """
    Convert a time-scaled phylogenetic tree into a genetic-distance-based tree.

    This function simulates the genetic distances (branch lengths) by:
    - randomly choosing a sequence alignment length,
    - randomly selecting a substitution rate,
    - simulating the number of mutations using a Poisson distribution.

    Args:
        time_tree_nwk (str): Path to the input Newick tree with time-based branch lengths.
        output_nwk (str): Path where the genetic-distance-based tree will be saved.

    Returns:
        tuple:
            - Tree: the ETE3 tree object with updated branch lengths.
            - int: the simulated alignment length used in mutation rate calculation.
    """
    time_tree = Tree(time_tree_nwk, format=1)

    # Randomly simulate alignment length and substitution rate
    alignment_length = np.random.randint(1000, 250000)
    r = np.random.uniform(5, 250)

    # Convert each branch length based on simulated mutation count
    for node in time_tree.traverse():
        if node.dist is not None:
            nb_mutations = np.random.poisson(node.dist * r)
            node.dist = nb_mutations / alignment_length

    # Save the new tree with genetic distances
    time_tree.write(format=1, outfile=output_nwk)
    return time_tree, alignment_length

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(
            "Usage: python3 time_to_genetic_distance.py "
            "input_tree.nwk output_genetic_tree.nwk [alignment_length.txt]"
        )
        sys.exit(1)

    input_nwk = sys.argv[1]
    output_nwk = sys.argv[2]
    length_file = sys.argv[3] if len(sys.argv) == 4 else None

    _, alignment_length = time_tree_to_genetic_tree(input_nwk, output_nwk)

    if length_file:
        with open(length_file, "w") as f:
            f.write(str(alignment_length))

