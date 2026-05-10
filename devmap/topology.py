
import networkx as nx
from copy import deepcopy
import numpy as np
import pandas as pd

def compress_single_child_internal_nodes(
    G: nx.DiGraph,
    attr_names=None,
    copy_graph=True,
):
    """
    Remove non-root internal nodes that have exactly one child by connecting
    each parent directly to that child.

    The new edge attributes are formed by summing the parent->node and
    node->child edge attributes for the requested attribute names.

    Parameters
    ----------
    G : nx.DiGraph
        Input directed graph.
    attr_names : iterable[str] | None
        Edge attribute names to sum. If None, sums all numeric attributes
        present on either edge.
    copy_graph : bool
        If True, work on a copy and return it. If False, mutate G in place.

    Returns
    -------
    nx.DiGraph
        The compressed graph.

    Notes
    -----
    A node is removed if:
      - it is not the root
      - it has out_degree == 1
      - it has in_degree >= 1
    Leaves are not removed.
    """

    if copy_graph:
        G = G.copy()

    root = [n for n, d in G.in_degree() if d == 0]
    if len(root) != 1:
        raise ValueError(f"Graph must have exactly one root node, found {len(root)}")
    root = root[0]

    def summed_attrs(attrs1, attrs2):
        out = {}

        if attr_names is None:
            keys = set(attrs1) | set(attrs2)
        else:
            keys = set(attr_names)

        for k in keys:
            v1 = attrs1.get(k, 0)
            v2 = attrs2.get(k, 0)

            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                out[k] = v1 + v2
            elif k in attrs1 and k not in attrs2:
                out[k] = deepcopy(v1)
            elif k in attrs2 and k not in attrs1:
                out[k] = deepcopy(v2)
            elif v1 == 0:
                out[k] = deepcopy(v2)
            elif v2 == 0:
                out[k] = deepcopy(v1)
            else:
                raise TypeError(
                    f"Cannot sum non-numeric edge attribute '{k}': {v1!r}, {v2!r}"
                )

        return out

    changed = True
    while changed:
        changed = False

        # Snapshot since we'll mutate during iteration
        for node in list(G.nodes):
            if node == root or node not in G:
                continue

            parents = list(G.predecessors(node))
            children = list(G.successors(node))

            # internal node with exactly one child
            if len(parents) >= 1 and len(children) == 1:
                child = children[0]

                # avoid creating self-loops like p -> node -> p
                if child == node:
                    continue

                child_edge_attrs = G.edges[node, child]

                new_edges = []
                for parent in parents:
                    if parent == child:
                        continue

                    parent_edge_attrs = G.edges[parent, node]
                    merged = summed_attrs(parent_edge_attrs, child_edge_attrs)

                    # If parent->child already exists, sum again into it
                    if G.has_edge(parent, child):
                        existing = dict(G.edges[parent, child])
                        merged_total = summed_attrs(existing, merged)
                        new_edges.append((parent, child, merged_total))
                    else:
                        new_edges.append((parent, child, merged))

                # Add replacement edges before removing node
                for u, v, attrs in new_edges:
                    G.add_edge(u, v, **attrs)

                G.remove_node(node)
                changed = True
                break

    return G

def marked_branches_by_depth(G, bins=np.arange(0, 10, 0.5), time_key="time"):
    """
    Fraction of branches marked by an edit by depth/time bin.

    Counts edges, not divisions:
    - Every observed edge in the collapsed tree is a marked branch.
    - A multifurcation with k children contributes k-2 hidden unmarked branches.
    - Observed edge time = midpoint of parent and child times.
    - Hidden branch time = midpoint of node time and mean(child times).

    Returns a DataFrame with per-bin marked/total fractions.
    """

    marked_times = []
    hidden_times = []

    # observed branches = actual edges in collapsed tree
    for parent, child in G.edges:
        tp = float(G.nodes[parent][time_key])
        tc = float(G.nodes[child][time_key])
        marked_times.append(0.5 * (tp + tc))
        #marked_times.append(tp)

    # hidden branches from multifurcations
    for n in G.nodes:
        children = list(G.successors(n))
        k = len(children)
        if k > 2:
            tn = float(G.nodes[n][time_key])
            child_times = np.array([float(G.nodes[c][time_key]) for c in children])
            t_hidden = 0.5 * (tn + child_times.mean())
            T = child_times.mean()
            # k-2 hidden branches, spaced on a log(cell-count) scale
            for j in range(1, k - 1):   # j = 1, ..., k-2
                tj = tn + (np.log(j + 1) / np.log(k)) * (T - tn)
                hidden_times.append(tj)

    marked_counts, edges = np.histogram(marked_times, bins=bins)
    hidden_counts, _ = np.histogram(hidden_times, bins=bins)
    total_counts = marked_counts + hidden_counts

    fraction = np.divide(
        marked_counts,
        total_counts,
        out=np.full_like(marked_counts, np.nan, dtype=float),
        where=total_counts > 0
    )

    return pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "bin_center": 0.5 * (edges[:-1] + edges[1:]),
        "marked_branches": marked_counts,
        "hidden_branches": hidden_counts,
        "total_branches": total_counts,
        "pct_marked": fraction * 100,
    })