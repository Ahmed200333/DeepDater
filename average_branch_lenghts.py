#!/usr/bin/env python3
import os
import glob
from ete3 import Tree

def average_branch_length(dir_path):
    pattern = os.path.join(dir_path, "BD_tree*_genetic_prefixed.nwk")
    files = glob.glob(pattern)
    total_length = 0.0
    total_edges = 0

    for fpath in files:
        tree = Tree(fpath, format=1)
        for node in tree.traverse():
            if not node.is_root():
                total_length += node.dist
                total_edges += 1

    if total_edges == 0:
        return 0.0
    return total_length / total_edges

if __name__ == "__main__":
    dir_path = "processed/0"
    avg = average_branch_length(dir_path)
    print("Longueur de branche moyenne sur tous les arbres :", avg)
