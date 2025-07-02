import numpy as np
import pandas as pd
import random
from ete3 import Tree
from treesimulator.generator import generate
from treesimulator import save_forest
from treesimulator.mtbd_models import BirthDeathModel


def simulate_n_BD_trees_with_dates(min_tips, max_tips, nwk_path, csv_path):
    """
    Simulates a single Birth-Death phylogenetic tree with temporal information
    and exports:
      - the tree in Newick format
      - a CSV with the sampling dates of all leaves (tips).

    Parameters:
    - min_tips: Minimum number of tips (int)
    - max_tips: Maximum number of tips (int)
    - nwk_path: Output path for the Newick tree file (str)
    - csv_path: Output path for the leaf date CSV (str)
    """
    # Sample parameters for the Birth-Death model
    sampling_probability = np.random.uniform(0.05, 0.75)
    infection_duration = np.random.uniform(1, 10)
    psi = 1 / infection_duration
    R0 = np.random.uniform(1, 10)
    birth_rate = R0 * psi

    # Create BD model and simulate one tree
    bd_model = BirthDeathModel(p=sampling_probability, la=birth_rate, psi=psi)
    [tree], _, _ = generate(bd_model, min_tips=min_tips, max_tips=max_tips)

    # Save tree to Newick file
    save_forest([tree], nwk_path)

    # Assign a root date randomly
    root_date = random.randint(1900, 2000)

    # Compute tip dates
    dates = []
    for leaf in tree.iter_leaves():
        distance_to_root = leaf.get_distance(tree.get_tree_root())
        leaf_date = root_date + distance_to_root
        dates.append({"Leaf": leaf.name, "Date": leaf_date})

    # Save tip dates to CSV
    df = pd.DataFrame(dates)
    df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Simulate a dated Birth-Death tree and export leaf dates.")
    parser.add_argument('--nwk', default='BD_tree.nwk', type=str, help="Output Newick file path")
    parser.add_argument('--csv', default='leaf_dates.csv', type=str, help="Output CSV file for leaf dates")
    parser.add_argument('--min_tips', type=int, required=True, help="Minimum number of tips")
    parser.add_argument('--max_tips', type=int, required=True, help="Maximum number of tips")

    args = parser.parse_args()
    simulate_n_BD_trees_with_dates(args.min_tips, args.max_tips, args.nwk, args.csv)

