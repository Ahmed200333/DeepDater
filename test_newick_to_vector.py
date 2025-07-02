import numpy as np
import ete3
from scipy import *
import jax.numpy as jnp
from jax import jit
import jax
from ete3 import Tree

# Example tree incidence matrices (for testing/debugging)
m_tree_test = np.array([[0, 1, 3, 2],
                        [-1, 0, 2, 0],
                        [-3, -2, 0, 0],
                        [-2, 0, 0, 0]])

m_tree_test_2 = np.array([[0, 3, 4, 5, 5, 4, 7, 6, 5],
                          [-3, 0, 0, 2, 2, 1, 0, 0, 0],
                          [-4, 0, 0, 0, 0, 0, 3, 2, 1],
                          [-5, -2, 0, 0, 0, 0, 0, 0, 0],
                          [-5, -2, 0, 0, 0, 0, 0, 0, 0],
                          [-4, -1, 0, 0, 0, 0, 0, 0, 0],
                          [-7, 0, -3, 0, 0, 0, 0, 0, 0],
                          [-6, 0, -2, 0, 0, 0, 0, 0, 0],
                          [-5, 0, -1, 0, 0, 0, 0, 0, 0]])

m_tree_test_3 = np.array([[0, 1, 2, 3],
                          [-1, 0, 0, 2],
                          [-2, 0, 0, 0],
                          [-3, -2, 0, 0]])


def matrix_to_dict(m: np.array):
    """
    Convert a tree incidence matrix into a structured dictionary describing node relationships.

    Args:
        m (np.array): A square matrix representing the tree in antisymmetric form.

    Returns:
        dict: A dictionary where each node stores its parent, children, and distances.
    """
    tree_dict = {}

    # Initialize each node in the dictionary
    for i in range(m.shape[0]):
        tree_dict[f"N{i}"] = {
            "parent": None,
            "direct_children": [],
            "distances_to_descendants": {},
            "distance_to_parent": 1.0
        }

    # Infer parent-child relationships
    for j in range(m.shape[0]):
        potential_parents = []
        for k in range(m.shape[0]):
            if m[j][k] < 0:
                potential_parents.append((m[j][k], k))

        if len(potential_parents) != 0:
            parent_value, parent_index = max(potential_parents, key=lambda x: x[0])
            tree_dict[f"N{j}"]["parent"] = f"N{parent_index}"
            tree_dict[f"N{parent_index}"]["direct_children"].append(f"N{j}")
            tree_dict[f"N{parent_index}"]["distances_to_descendants"][f"N{j}"] = m[parent_index][j]
            tree_dict[f"N{j}"]["distance_to_parent"] = -parent_value

    return tree_dict


def dict_to_ete3_tree(tree_dict: dict):
    """
    Convert a structured tree dictionary to an ETE3 tree.

    Args:
        tree_dict (dict): Tree in dictionary format with parent-child relationships.

    Returns:
        ete3.Tree: ETE3 tree object.
    """
    nodes = {name: Tree(name=name) for name in tree_dict}
    root = None

    for node_name, info in tree_dict.items():
        parent_name = info["parent"]
        if parent_name is None:
            root = nodes[node_name]
        else:
            parent_node = nodes[parent_name]
            child_node = nodes[node_name]
            parent_node.add_child(child_node, dist=info["distance_to_parent"])
    return root


def matrix_to_ete3_tree(m: np.array):
    """
    Convert an incidence matrix into an ETE3 tree object.

    Args:
        m (np.array): Antisymmetric matrix representing the tree.

    Returns:
        ete3.Tree: ETE3 tree reconstructed from the matrix.
    """
    return dict_to_ete3_tree(matrix_to_dict(m))


def ete3_tree_to_dict(t):
    """
    Convert an ETE3 tree to a structured dictionary with node metadata.

    Args:
        t (ete3.Tree): ETE3 tree object.

    Returns:
        dict: Tree dictionary with relationships and distances.
    """
    tree_dict = {}
    internal_node_counter = 0

    for node in t.traverse():
        if node.name == "":
            node.name = f"internal_{internal_node_counter}"
            internal_node_counter += 1

    for node in t.traverse():
        node_name = node.name
        parent_name = node.up.name if node.up else None
        distance = node.dist
        distance_to_root = distance
        current_node = node.up

        while current_node:
            distance_to_root += current_node.dist
            current_node = current_node.up

        tree_dict[node_name] = {
            "parent": parent_name,
            "direct_children": [child.name for child in node.children],
            "distances_to_descendants": {child.name: child.dist for child in node.children},
            "distance_to_parent": distance,
            "distance_to_root": distance_to_root
        }

    return tree_dict


def dict_to_matrix(d):
    """
    Convert a tree dictionary into an antisymmetric incidence matrix.

    Args:
        d (dict): Tree dictionary representation.

    Returns:
        np.array: Antisymmetric incidence matrix.
    """
    sorted_nodes = sorted(d.keys(), key=lambda x: d[x]["distance_to_root"], reverse=True)
    n = len(d)
    node_index = {name: i for i, name in enumerate(sorted_nodes)}
    m = np.zeros((n, n))

    for node_name in sorted_nodes:
        node_idx = node_index[node_name]
        parent_name = d[node_name]["parent"]

        if parent_name is not None:
            parent_idx = node_index[parent_name]
            distance = d[node_name]["distance_to_parent"]

            m[node_idx][parent_idx] = -distance
            m[parent_idx][node_idx] = distance

            current_parent = parent_name
            cumulative_distance = distance
            while current_parent is not None:
                grandparent_name = d[current_parent]["parent"]
                if grandparent_name is not None:
                    grandparent_idx = node_index[grandparent_name]
                    cumulative_distance += d[current_parent]["distance_to_parent"]
                    m[node_idx][grandparent_idx] = -cumulative_distance
                    m[grandparent_idx][node_idx] = cumulative_distance
                current_parent = grandparent_name

    return m


def ete3_to_matrix(t):
    """
    Convert an ETE3 tree to its matrix representation.

    Args:
        t (ete3.Tree): ETE3 tree.

    Returns:
        np.array: Antisymmetric matrix representation.
    """
    tree_dict = ete3_tree_to_dict(t)
    return dict_to_matrix(tree_dict)


@jit
def matrix_to_optimized_jax_vector(m):
    """
    Compress an antisymmetric matrix into a JAX-compatible vector (lower triangle only).

    Args:
        m (jax.numpy.ndarray): Antisymmetric matrix.

    Returns:
        jax.numpy.ndarray: Compressed lower-triangle vector.
    """
    n = m.shape[0]
    lower_tri = jnp.tril(m, k=-1)
    return lower_tri[jnp.tril_indices(n, k=-1)]


def jax_optimized_vector_to_matrix(v, n):
    """
    Reconstruct a square antisymmetric matrix from a lower-triangle vector.

    Args:
        v (jax.numpy.ndarray): Lower-triangle compressed vector.
        n (int): Matrix dimension.

    Returns:
        jax.numpy.ndarray: Reconstructed antisymmetric matrix.
    """
    m = jnp.zeros((n, n))
    row_idx, col_idx = jnp.tril_indices(n, k=-1)
    m = m.at[(row_idx, col_idx)].set(v)
    m = m.at[(col_idx, row_idx)].set(-v)
    return m


jax_optimized_vector_to_matrix = jit(jax_optimized_vector_to_matrix, static_argnums=1)


def newick_to_jax_vector(newick_file_name):
    """
    Load a Newick tree and return its compressed antisymmetric vector representation.

    Args:
        newick_file_name (str): Path to Newick file.

    Returns:
        jax.numpy.ndarray: Compressed vector of the tree.
    """
    tree = Tree(newick_file_name, format=1, quoted_node_names=True)
    tree = name_internal_nodes(tree)
    mat = ete3_to_matrix(tree)
    return matrix_to_optimized_jax_vector(mat)


def jax_vector_to_newick(vect):
    """
    Convert a compressed vector into a Newick tree. [Note: returns nothing.]

    Args:
        vect (jax.numpy.ndarray): Compressed antisymmetric vector.
    """
    mat_size = get_matrix_size_from_vector(vect)
    mat = jax_optimized_vector_to_matrix(vect, mat_size)
    tree = matrix_to_ete3_tree(mat)
    # Output not returned


def ete3_tree_to_jax_vector(t):
    """
    Convert an ETE3 tree directly to a compressed JAX vector.

    Args:
        t (ete3.Tree): Input tree.

    Returns:
        jax.numpy.ndarray: Compressed vector.
    """
    mat = ete3_to_matrix(t)
    return matrix_to_optimized_jax_vector(mat)


def get_matrix_size_from_vector(v):
    """
    Infer original matrix size from a compressed vector length.

    Args:
        v (np.ndarray): Compressed vector.

    Returns:
        int: Original matrix size.
    """
    x = v.shape[0]
    n_float = (1 + np.sqrt(1 + 8 * x)) / 2
    return int(np.round(n_float))


def get_number_of_nodes(t):
    """
    Count unique nodes in an ETE3 tree.

    Args:
        t (ete3.Tree): Input tree.

    Returns:
        int: Number of unique nodes.
    """
    return len(set(node.name for node in t.traverse()))


def name_internal_nodes(tree):
    """
    Assign unique names to unnamed internal nodes in a tree.

    Args:
        tree (ete3.Tree): Input tree.

    Returns:
        ete3.Tree: Tree with named internal nodes.
    """
    internal_count = 0
    for node in tree.traverse():
        if not node.is_leaf() and node.name == "":
            node.name = f"internal_{internal_count}"
            internal_count += 1
    return tree


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        sys.stderr.write(
            "Usage: python3 test_newick_to_vector.py <input_tree.nexus> <output_vector.npy>\n"
        )
        sys.exit(1)

    input_tree = sys.argv[1]
    output_vec = sys.argv[2]

    vect = newick_to_jax_vector(input_tree)
    np.save(output_vec, vect)
    print(f"Vector saved to '{output_vec}'")

