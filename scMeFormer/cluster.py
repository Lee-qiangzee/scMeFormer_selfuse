import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components

def main():
    y_file = 'HCC/y.npz'
    data = np.load(y_file, allow_pickle=True)
    keys = list(data.keys())

    bin_size = 10000
    fraction_list = []
    for chr_name in keys:
        y_chr = data[chr_name]               # shape = (n_sites, n_cells)
        n_sites, n_cells = y_chr.shape
        n_bins = n_sites // bin_size
        if n_bins == 0:
            continue

        fractions = np.zeros((n_bins, n_cells), dtype=float)
        for b in range(n_bins):
            block = y_chr[b*bin_size:(b+1)*bin_size, :]  # (bin_size, n_cells)
            fractions[b, :] = (block == 1).sum(axis=0) / bin_size

        fraction_list.append(fractions.T)

    cell_features = np.concatenate(fraction_list, axis=1)

    k = 5
    A = kneighbors_graph(
        cell_features,
        n_neighbors=k,
        mode='connectivity',
        include_self=False
    ).tocsr()

    A_mutual = A.multiply(A.T)

    n_components, labels = connected_components(
        csgraph=A_mutual,
        directed=False,
        return_labels=True
    )

    txt_file = 'cell_clusters.txt'
    with open(txt_file, 'w') as f:
        for comp in range(n_components):
            members = np.where(labels == comp)[0].tolist()
            # 簇编号从 1 开始
            f.write(f"{members}\n")

if __name__ == '__main__':
    main()
