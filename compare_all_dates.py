#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from ete3 import Tree
import re
import os

def parse_arguments():
    if len(sys.argv) != 5:
        sys.stderr.write(
            "Usage: python3 compare_all_dates.py "
            "<true_time_tree.nwk> <leaf_dates.csv> <lsd2_output.nexus> <out_metrics.txt>\n"
        )
        sys.exit(1)
    return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def name_internal_nodes(tree: Tree) -> Tree:
    """Assigns unique names to unnamed internal nodes."""
    count = 0
    for node in tree.traverse():
        if not node.is_leaf() and node.name.strip() == "":
            node.name = f"node_{count}"
            count += 1
    return tree

def get_ordered_nodes(tree: Tree) -> list:
    """
    Returns a list of node names sorted by decreasing distance from the root.
    """
    node_distances = {node.name: node.get_distance(tree) for node in tree.traverse()}
    sorted_nodes = sorted(node_distances, key=node_distances.get, reverse=True)
    print(f"Number of nodes: {len(sorted_nodes)}")
    return sorted_nodes

def build_true_vector(time_tree_file: str, leaf_csv_file: str, node_order: list) -> np.ndarray:
    """
    Builds the ground-truth date vector from the known time tree and leaf CSV.
    """
    tree = name_internal_nodes(Tree(time_tree_file))
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

def extract_newick_from_nexus(nexus_path: str) -> tuple:
    """
    Extracts the Newick string and associated node dates from an LSD2 .nexus file.
    """
    with open(nexus_path, "r") as f:
        text = f.read()

    node_dates = {}

    def repl_leaf(m):
        name = m.group(1)
        date_str = m.group(2)
        branch_len = m.group(3)
        date_float = float(date_str)
        new_name = f"{name}_pred"
        node_dates[new_name] = date_float
        return f"{new_name}:{branch_len}"

    def repl_internal(m):
        date_str = m.group(1)
        branch_len = m.group(2)
        date_float = float(date_str)
        new_name = f"node_pred_{len(node_dates)}"
        node_dates[new_name] = date_float
        return f"){new_name}:{branch_len}"

    text = re.sub(r"(leaf\d+)\[\&date=\"?([\d\.\-]+)\"?\]:(\d*\.?\d+)", repl_leaf, text)
    text = re.sub(r"\)\[\&date=\"?([\d\.\-]+)\"?\]:(\d*\.?\d+)", repl_internal, text)
    text = re.sub(r"\[\&date=.*?\]", "", text)

    # Extract Newick string
    lines = text.splitlines()
    newick_str = ""
    collecting = False
    for line in lines:
        if not collecting:
            m = re.match(r"\s*tree\s+\d+\s*=\s*(.*)", line, re.IGNORECASE)
            if m:
                rest = m.group(1).strip()
                if ";" in rest:
                    return rest[:rest.index(";")+1], node_dates
                newick_str = rest + "\n"
                collecting = True
        else:
            if ";" in line:
                newick_str += line[:line.index(";")+1]
                break
            else:
                newick_str += line + "\n"

    return newick_str.strip(), node_dates

def build_pred_vector(nexus_path: str, node_order: list, leaf_csv_file: str) -> np.ndarray:
    """
    Builds the predicted date vector from the LSD2 .nexus output.
    """
    newick_str, node_dates = extract_newick_from_nexus(nexus_path)

    try:
        tree = Tree(newick_str, format=1, quoted_node_names=True)
    except:
        tree = Tree(newick_str, format=1)
    tree = name_internal_nodes(tree)

    name_to_date = {
        node.name: node_dates[node.name]
        for node in tree.traverse()
        if node.name in node_dates
    }

    node_order = get_ordered_nodes(tree)
    root = tree.get_tree_root()

    vec = [name_to_date.get(node_name, 0.0) for node_name in node_order]

    # If root date is missing, try recovering from the log file (tMRCA)
    if vec[-1] == 0.0:
        try:
            log_path = os.path.join(
                os.path.dirname(nexus_path),
                os.path.basename(nexus_path).replace(".date.nexus", "")
            )
            with open(log_path, "r") as f:
                for line in f:
                    match = re.search(r"tMRCA\s+(\d+\.\d+)", line)
                    if match:
                        tmrca_date = float(match.group(1))
                        vec[-1] = tmrca_date
                        print(f"[INFO] Root date recovered from tMRCA: {tmrca_date}")
                        break
        except Exception as e:
            print(f"[WARN] Could not retrieve tMRCA from {log_path}: {e}")

    print(f"Predicted vector size: {len(vec)}")
    print(f"Predicted vector: {vec}")
    return np.array(vec)

def compute_metrics(true_vec: np.ndarray, pred_vec: np.ndarray, node_order: list) -> tuple:
    """
    Computes RRE and NRMSE_internal metrics between the true and predicted vectors.
    """
    t_root_true = true_vec[-1]
    t_root_pred = pred_vec[-1]
    H = np.max(true_vec) - np.min(true_vec)
    print(f"Tree height H: {H}")
    RRE = abs(t_root_pred - t_root_true) / H

    internal_idxs = [
        i for i, name in enumerate(node_order)
        if name.startswith("node")
    ]
    print(f"Number of internal nodes: {len(internal_idxs)}")

    if len(internal_idxs) == 0:
        NRMSE = 0.0
    else:
        sum_err = sum(
            abs(pred_vec[i] - true_vec[i]) / H
            for i in internal_idxs
        )
        NRMSE = sum_err / len(internal_idxs)

    print(f"RRE = {RRE:.6f}, NRMSE_internal = {NRMSE:.6f}")
    return RRE, NRMSE

def write_metrics(path: str, RRE: float, NRMSE: float) -> None:
    """
    Saves the computed metrics to a text file.
    """
    with open(path, "w") as f:
        f.write(f"RRE\t{RRE:.6f}\n")
        f.write(f"NRMSE_internal\t{NRMSE:.6f}\n")
        f.write("Mean_RRE\t0.0\n")
        f.write("Mean_NRMSE_internal\t0.0\n")

def main(time_tree: str, leaf_csv: str, nexus_out: str, out_txt: str):
    ete_tree = name_internal_nodes(Tree(time_tree))
    node_order = get_ordered_nodes(ete_tree)
    true_vec = build_true_vector(time_tree, leaf_csv, node_order)
    pred_vec = build_pred_vector(nexus_out, node_order, leaf_csv)
    RRE, NRMSE = compute_metrics(true_vec, pred_vec, node_order)
    write_metrics(out_txt, RRE, NRMSE)
    print(f" RRE = {RRE:.6f}, NRMSE_internal = {NRMSE:.6f}")

if __name__ == "__main__":
    time_tree, leaf_csv, nexus_out, out_txt = parse_arguments()
    main(time_tree, leaf_csv, nexus_out, out_txt)

