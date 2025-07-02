import numpy as np
from ete3 import Tree
from test_newick_to_vector import newick_to_jax_vector
from csv_to_dates_vector import (
    build_complete_date_vector,
    get_ordered_nodes,
    csv_to_dates_vector,
    name_internal_nodes,
)
from tensorflow import keras
import os

def restore_dates_on_tree(tree_path: str, csv_path: str, model_path: str, output_path: str) -> None:
    """
    Applies a trained model to infer internal node dates of a phylogenetic tree,
    using partial leaf dates as input.

    Args:
        tree_path (str): Path to the input Newick file (.nwk) of the genetic tree.
        csv_path (str): Path to the CSV file containing known leaf dates.
        model_path (str): Path to the trained Keras model file.
        output_path (str): Path to save the output dated Newick tree with internal dates.
    """
    tree = name_internal_nodes(Tree(tree_path, format=1))
    ordered_nodes = get_ordered_nodes(tree)

    # Convert tree to feature vector
    X = newick_to_jax_vector(tree_path)
    mean_len = X[X > 0].mean() if np.any(X > 0) else 1
    X_norm = X / mean_len

    # Prepare partial date vector (normalized)
    y_partial = csv_to_dates_vector(csv_path, ordered_nodes)
    max_date = np.max(y_partial)
    y_partial_norm = 1 - (max_date - y_partial)

    # Retrieve max lengths from training dataset
    max_len_X = np.load("final_dataset.npz")["X"].shape[1]
    max_len_y = np.load("final_dataset.npz")["y_partial"].shape[1]

    # Pad inputs
    X_padded = np.pad(X_norm, (0, max_len_X - len(X_norm)), constant_values=0)
    y_partial_padded = np.pad(y_partial_norm, (0, max_len_y - len(y_partial_norm)), constant_values=0)

    # Model input
    X_input = np.concatenate([X_padded, y_partial_padded]).reshape(1, -1)

    # Load model and predict
    model = keras.models.load_model(model_path, compile=False)
    y_pred_norm = model.predict(X_input)[0]

    # Denormalize predicted dates
    y_pred = max_date - (1 - y_pred_norm)

    # Merge known leaf dates and predicted internal dates
    final_dates = []
    for yp, yc in zip(y_partial, y_pred):
        final_dates.append(yp if yp > 0 else yc)

    # Assign dates as node features
    for name, date in zip(ordered_nodes, final_dates):
        tree.search_nodes(name=name)[0].add_features(date=date)

    # Save output tree
    tree.write(outfile=output_path, format=5, format_root_node=True, features=["date"])
    print(f"\nDated tree saved to: {output_path}")

def ask_path(prompt: str, ext: str) -> str:
    """
    Prompts user for a file path and checks its validity.

    Args:
        prompt (str): Prompt message.
        ext (str): Expected file extension.

    Returns:
        str: Validated file path.
    """
    path = input(prompt).strip()
    while not os.path.isfile(path) or not path.endswith(ext):
        print(" Invalid file. Please try again.")
        path = input(prompt).strip()
    return path

if __name__ == "__main__":
    print("Phylogenetic Tree Dating Tool")
    tree_path = ask_path(" Path to your genetic tree (.nwk): ", ".nwk")
    csv_path = ask_path(" Path to your leaf dates CSV (.csv): ", ".csv")
    model_path = input(" Path to your trained model (.keras) [default: deepl_dating_model.keras]: ").strip()
    if not model_path:
        model_path = "deepl_dating_model.keras"
    output_path = input(" Output file name (.nwk) [default: dated_tree.nwk]: ").strip()
    if not output_path:
        output_path = "dated_tree.nwk"

    restore_dates_on_tree(tree_path, csv_path, model_path, output_path)

