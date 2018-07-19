import os
import matplotlib
# matplotlib.use('Agg')
import numpy as np


def visualize_data_partition(X,
                             fig_size=(10, 5),
                             # cmap="husl",
                             cmap="Set2",
                             title=None,
                             y_label='N',
                             x_label='D',
                             color_map_ids=None,
                             show_fig=True,
                             output=None):
    """
    Coloring X, a partitioning of an NxD matrix containing the integer ids of the partitions each entry belongs to
    """
    import seaborn as sns
    from matplotlib.colors import ListedColormap
    import matplotlib
    if not show_fig:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_pdf import PdfPages

    sns.reset_orig()  # get default matplotlib styles back

    N, D = X.shape

    cmap_colors = np.unique(X)
    cmpa_colors_inv_map = {c: i for i, c in enumerate(cmap_colors)}
    clrs = sns.color_palette(cmap, n_colors=len(cmap_colors))
    sns_cmap = ListedColormap(clrs.as_hex())

    X_trans = np.array(X)
    for n in range(N):
        for d in range(D):
            X_trans[n, d] = cmpa_colors_inv_map[X[n, d]]

    print('using colors', cmap_colors)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.matshow(X_trans, cmap=sns_cmap, aspect=D / N)
    ax.legend([mpatches.Patch(color=clrs[i]) for i, b in enumerate(cmap_colors)],
              [color_map_ids[b] for b in cmap_colors], loc='center left', bbox_to_anchor=(1, 0.5))

    if title:
        plt.title(title)

    if y_label:
        ax.set_ylabel(y_label)

    if x_label:
        ax.set_xlabel(x_label)

    # plt.tight_layout()

    if output:
        pp = PdfPages(output)
        pp.savefig(fig, bbox_inches='tight')
        pp.close()

    if show_fig:
        plt.show()


def visualize_histogram(X,
                        fig_size=(10, 5),
                        cmap="husl",
                        title=None,
                        bins=150,
                        density=True,
                        y_label='density',
                        x_label='samples',
                        show_fig=True,
                        output=None):
    """
    Plotting the density for a univariate data X
    """
    import seaborn as sns
    from matplotlib.colors import ListedColormap
    import matplotlib
    if not show_fig:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_pdf import PdfPages

    # sns.reset_orig()  # get default matplotlib styles back

    fig, ax = plt.subplots(figsize=fig_size)

    #
    # missing values?
    X = X[~np.isnan(X)]

    ax.hist(X, density=density, bins=bins)

    if title:
        plt.title(title)

    if y_label:
        ax.set_ylabel(y_label)

    if x_label:
        ax.set_xlabel(x_label)

    plt.tight_layout()

    if output:
        pp = PdfPages(output)
        pp.savefig(fig)
        pp.close()

    if show_fig:
        plt.show()


def reorder_data_partitions(X, cmap="husl"):

    import seaborn as sns
    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scipy.cluster.hierarchy as hac

    Z = hac.linkage(X, method="single")
    D = hac.dendrogram(Z)
    plt.close()
    return np.array([int(i) for i in D['ivl']])


from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.Inference import likelihood


def approximate_density(dist_node, X, bins=100):
    if dist_node.type.meta_type == MetaType.DISCRETE:
        x = np.array([i for i in range(int(np.nanmin(X)), int(np.nanmax(X)) + 1)])
    else:
        x = np.linspace(np.nanmin(X), np.nanmax(X), bins)
    x = x.reshape(-1, 1)
    y = likelihood(dist_node, x)
    return x[:, 0], y[:, 0]


def approximate_density_d(dist_node, X, d, bins=100, meta_type=None):
    X_marg = np.copy(X)

    if meta_type == MetaType.DISCRETE:
        x = np.array([i for i in range(int(np.nanmin(X[:, d])), int(np.nanmax(X[:, d])) + 1)])
    else:
        x = np.linspace(np.nanmin(X[:, d]), np.nanmax(X[:, d]), bins)
    X_marg = np.zeros((x.shape[0], X.shape[1]))
    X_marg[:] = np.nan
    X_marg[:, d] = x
    x = x.reshape(-1, 1)
    y = likelihood(dist_node, X_marg)
    return x[:, 0], y[:, 0]


def plot_distributions_fitting_data(data,
                                    dist_nodes,
                                    f_id,
                                    type_leaf_id,
                                    bins=100,
                                    weight_scaled=None,
                                    show_fig=False,
                                    save_fig=None,
                                    cmap=None):
    import matplotlib
    if not show_fig:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    from matplotlib.colors import ListedColormap

    sns.reset_orig()  # get default matplotlib styles back
    clrs = sns.color_palette(cmap, n_colors=len(dist_nodes) + 1)
    sns_cmap = ListedColormap(clrs.as_hex())

    n_samples = data.shape[0]
    bin_lims = np.linspace(0, 1, bins + 1)
    bin_centers = 0.5 * (bin_lims[:-1] + bin_lims[1:])
    bin_widths = bin_lims[1:] - bin_lims[:-1]

    fig, ax = plt.subplots(1, 1)
    area = 0
    l_hists = [None for l in dist_nodes]
    l_bins = [None for l in dist_nodes]
    m_data = []
    all_ids = set()
    for j, l in enumerate(dist_nodes):
        pdf_x, pdf_y = approximate_density(l, data[:, f_id], bins=bins)

        if weight_scaled:
            pdf_y = pdf_y * weight_scaled[j]

        # if l.type.meta_type == MetaType.DISCRETE:
        #     ax.bar(pdf_x, pdf_y, label="leaf {}: {}".format(l.id,
        #                                                     l.name), alpha=0.4)
        # elif l.type.meta_type == MetaType.REAL:
        ax.plot(pdf_x, pdf_y, label="paramform: {}".format(l.name), color=clrs[j])
        #
        # drawing also the data, coloured by the membership
    # à
    #     if len(l.row_ids) > 0:
    #         hist, _bins = np.histogram(data[l.row_ids, f_id], bins=bins)
    #         area += (np.diff(_bins) * hist).sum()
    #         l_hists[j] = hist
    #         l_bins[j] = _bins
    # for j, l in enumerate(dist_nodes):
    #     if len(l.row_ids) > 0:
    #         l_hists[j] = l_hists[j] / area
    #         ax.bar(l_bins[j][:-1] + np.diff(l_bins[j]) / 2, l_hists[j], align='center', alpha=0.4)
    #########################################################################################
        x = data[l.row_ids, f_id]
        # print('LEN x', f_id, l.name, len(l.row_ids))
        miss_vals = np.isnan(x)
        x = x[~miss_vals]
        m_data.append(x)
        all_ids.update(np.array(l.row_ids)[~miss_vals])
    rem_ids = list(sorted(set(np.arange(n_samples)) - all_ids))
    x = data[rem_ids, f_id]
    miss_vals = np.isnan(x)
    x = x[~miss_vals]
    m_data.append(x)
    ax.hist(m_data, bins, histtype='bar', density=True, stacked=True, color=clrs)

    ax.legend()
    plt.title('Feature {} leaf {} [{} samples]'.format(f_id, type_leaf_id, len(all_ids)))

    if show_fig:
        plt.show()

    if save_fig:
        pp = PdfPages(save_fig)
        pp.savefig(fig)
        pp.close()

    plt.close()


from spn.structure.Base import get_nodes_by_type, compute_leaf_global_mix_weights, compute_partition_id_map, Product, Sum
from spn.structure.leaves.parametric.Parametric import Parametric
from spn.structure.leaves.typedleaves.TypedLeaves import TypeLeaf

TEXT_Y_POS_PERC = 0.01


def plot_mixture_components_fitting_data(spn,
                                         X,
                                         bins=100,
                                         threshold=0.01,
                                         show_fig=False,
                                         save_fig=None,
                                         cmap=None):
    import matplotlib
    if not show_fig:
        matplotlib.use('Agg')
    matplotlib.rcParams.update({'font.size': 24,
                                'axes.titlepad': 22})

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    from matplotlib.colors import ListedColormap

    N, D = X.shape

    leaf_weights = compute_leaf_global_mix_weights(spn)
    part_map = compute_partition_id_map(spn)
    uniq_partitions = list(sorted(set(part_map.values())))
    print('{} UNIQUE partitions {}'.format(len(uniq_partitions), uniq_partitions))

    sns.reset_orig()  # get default matplotlib styles back

    # sns_cmap = ListedColormap(clrs.as_hex())
    features_wine = ['fixed acidity',
                     'volatile acidity',
                     'citric acid',
                     'residual sugar',
                     'chlorides',
                     'free sulfur dioxide',
                     'total sulfur dioxide',
                     'density',
                     'pH',
                     'sulphates',
                     'alcohol',
                     'quality']
    for d in range(D):

        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        m_data = []
        all_ids = set()

        leaf_nodes_d = [l for l in get_nodes_by_type(spn, Parametric) if l.scope[0] == d]
        print('D', d, leaf_nodes_d)
        clrs = sns.color_palette(cmap, n_colors=len(leaf_nodes_d) + 1)
        act_clrs = []
        # wine_colors = [['#cccaa9', '#770a2d'], ['#770a2d', '#cccaa9'], ['#770a2d', '#cccaa9'],
        #                ['#cccaa9', '#770a2d'], ['#cccaa9', '#770a2d'], ['#770a2d', '#cccaa9'],
        #                ['#770a2d', '#cccaa9'], ['#770a2d', '#cccaa9'], ['#770a2d', '#cccaa9'],
        #                ['#770a2d', '#cccaa9'], ['#770a2d', '#cccaa9'], ['#770a2d', '#cccaa9']]
        for j, l in enumerate(leaf_nodes_d):

            w = leaf_weights[l]

            if w >= threshold:
                # if l.scope[0] == d:
                pdf_x, pdf_y = approximate_density(l, X[:, d], bins=bins)
                # pdf_y = pdf_y * w

                print(pdf_x, pdf_y)
                mode_id = np.argmax(pdf_y)
                mode_x = pdf_x[mode_id]
                mode_y = pdf_y[mode_id]

                part_id = part_map[l]

                # w_c = wine_colors[d][j % len(wine_colors[d])]
                ax.plot(pdf_x, pdf_y, label="{}: {:.4f}% ".format(l.__class__.__name__,
                                                                  # part_id,
                                                                  w),
                        color=clrs[j],
                        # color=w_c,
                        linewidth=3)
                # ax.text(mode_x, mode_y + mode_y * TEXT_Y_POS_PERC, '{}'.format(part_id))
                # act_clrs.append(w_c)
                act_clrs.append(clrs[j])

                x = X[l.row_ids, d]
            # print('LEN x', f_id, l.name, len(l.row_ids))
                miss_vals = np.isnan(x)
                x = x[~miss_vals]
                m_data.append(x)
                all_ids.update(np.array(l.row_ids)[~miss_vals])
        rem_ids = list(sorted(set(np.arange(N)) - all_ids))
        x = X[rem_ids, d]
        miss_vals = np.isnan(x)
        x = x[~miss_vals]
        m_data.append(x)
        act_clrs.append(clrs[-1])
        # ax.hist(m_data,
        #         bins,
        #         histtype='bar', density=True,
        #         stacked=True,
        #         # stacked=False,
        #         color=act_clrs, alpha=.5)

        ax.legend()
        plt.title('Wine: {}'.format(features_wine[d]))

        if show_fig:
            plt.show()

        if save_fig:
            output_path = os.path.join(save_fig, 'd{}-fit'.format(d))
            pp = PdfPages(output_path)
            pp.savefig(fig)
            pp.close()

        plt.close()


def plot_mixture_components_fitting_data_wine(spn,
                                              X,
                                              bins=100,
                                              threshold=0.01,
                                              show_fig=False,
                                              save_fig=None,
                                              cmap=None):
    import matplotlib
    if not show_fig:
        matplotlib.use('Agg')
    matplotlib.rcParams.update({'font.size': 24,
                                'axes.titlepad': 22})

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    from matplotlib.colors import ListedColormap

    N, D = X.shape

    leaf_weights = compute_leaf_global_mix_weights(spn)
    part_map = compute_partition_id_map(spn)
    uniq_partitions = list(sorted(set(part_map.values())))
    print('{} UNIQUE partitions {}'.format(len(uniq_partitions), uniq_partitions))

    sns.reset_orig()  # get default matplotlib styles back

    # sns_cmap = ListedColormap(clrs.as_hex())
    features_wine = ['fixed acidity',
                     'volatile acidity',
                     'citric acid',
                     'residual sugar',
                     'chlorides',
                     'free sulfur dioxide',
                     'total sulfur dioxide',
                     'density',
                     'pH',
                     'sulphates',
                     'alcohol',
                     'quality']
    for d in range(D):

        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        m_data = []
        all_ids = set()

        leaf_nodes_d = [l for l in get_nodes_by_type(spn, Parametric) if l.scope[0] == d]
        print('D', d, leaf_nodes_d)
        clrs = sns.color_palette(cmap, n_colors=len(leaf_nodes_d) + 1)
        act_clrs = []
        wine_colors = [['#cccaa9', '#770a2d'], ['#770a2d', '#cccaa9'], ['#770a2d', '#cccaa9'],
                       ['#cccaa9', '#770a2d'], ['#cccaa9', '#770a2d'], ['#770a2d', '#cccaa9'],
                       ['#770a2d', '#cccaa9'], ['#770a2d', '#cccaa9'], ['#770a2d', '#cccaa9'],
                       ['#770a2d', '#cccaa9'], ['#770a2d', '#cccaa9'], ['#770a2d', '#cccaa9']]
        for j, l in enumerate(leaf_nodes_d):

            w = leaf_weights[l]

            if w >= threshold:
                # if l.scope[0] == d:
                pdf_x, pdf_y = approximate_density(l, X[:, d], bins=bins)
                pdf_y = pdf_y * w

                print(pdf_x, pdf_y)
                mode_id = np.argmax(pdf_y)
                mode_x = pdf_x[mode_id]
                mode_y = pdf_y[mode_id]

                part_id = part_map[l]

                w_c = wine_colors[d][j % len(wine_colors[d])]
                ax.plot(pdf_x, pdf_y, label="{}: {:.4f}% ".format(l.__class__.__name__,
                                                                  # part_id,
                                                                  w),
                        # color=clrs[j],
                        color=w_c,
                        linewidth=3)
                # ax.text(mode_x, mode_y + mode_y * TEXT_Y_POS_PERC, '{}'.format(part_id))
                act_clrs.append(w_c)

                x = X[l.row_ids, d]
            # print('LEN x', f_id, l.name, len(l.row_ids))
                miss_vals = np.isnan(x)
                x = x[~miss_vals]
                m_data.append(x)
                all_ids.update(np.array(l.row_ids)[~miss_vals])
        rem_ids = list(sorted(set(np.arange(N)) - all_ids))
        x = X[rem_ids, d]
        miss_vals = np.isnan(x)
        x = x[~miss_vals]
        m_data.append(x)
        act_clrs.append(clrs[-1])
        ax.hist(m_data,
                bins,
                histtype='bar', density=True,
                stacked=True,
                # stacked=False,
                color=act_clrs, alpha=.5)

        ax.legend()
        plt.title('Wine: {}'.format(features_wine[d]))

        if show_fig:
            plt.show()

        if save_fig:
            output_path = os.path.join(save_fig, 'd{}-fit'.format(d))
            pp = PdfPages(output_path)
            pp.savefig(fig)
            pp.close()

        plt.close()


def plot_mixtures_fitting_multilevel(spn, X,
                                     meta_types,
                                     bins=100,
                                     threshold=0.01,
                                     show_fig=False,
                                     save_fig=None,
                                     cmap=None):

    partition_map = compute_partition_id_map(spn)
    #
    # getting all products
    sum_nodes = get_nodes_by_type(spn, Sum)
    #
    # getting also the root
    partition_nodes = [spn]
    for s in sum_nodes:
        partition_nodes.append(s)

    for node in partition_nodes:

        partition_vis_path = None
        if save_fig:
            partition_vis_path = os.path.join(save_fig, 'part-#{}'.format(node.id))
            os.makedirs(partition_vis_path, exist_ok=True)

        plot_mixture_components_fitting_data_recursive(node,
                                                       X,
                                                       meta_types,
                                                       partition_map,
                                                       bins=bins,
                                                       threshold=threshold,
                                                       show_fig=show_fig,
                                                       save_fig=partition_vis_path,
                                                       cmap=cmap)

    # mix_leaf_nodes = get_nodes_by_type(spn, TypeLeaf)

    # for leaf in mix_leaf_nodes:

    #     partition_vis_path = None
    #     if save_fig:
    #         partition_vis_path = os.path.join(save_fig, 'leaf-#{}'.format(node.id))
    #         os.makedirs(partition_vis_path, exist_ok=True)

    #     plot_mixture_components_fitting_data_recursive(node,
    #                                                    X,
    #                                                    meta_types,
    #                                                    bins=bins,
    #                                                    threshold=threshold,
    #                                                    show_fig=show_fig,
    #                                                    save_fig=partition_vis_path,
    #                                                    cmap=cmap)


def plot_mixture_components_fitting_data_recursive(node,
                                                   X,
                                                   meta_types,
                                                   part_map,
                                                   bins=100,
                                                   threshold=0.01,
                                                   show_fig=False,
                                                   save_fig=None,
                                                   cmap=None):
    import matplotlib
    if not show_fig:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    from matplotlib.colors import ListedColormap

    N, D = X.shape

    leaf_weights = compute_leaf_global_mix_weights(node)

    uniq_partitions = list(sorted(set(part_map.values())))
    print('{} UNIQUE partitions {}'.format(len(uniq_partitions), uniq_partitions))

    sns.reset_orig()  # get default matplotlib styles back

    # sns_cmap = ListedColormap(clrs.as_hex())
    D = node.scope
    for d in D:

        fig, ax = plt.subplots(1, 1)
        m_data = []
        all_ids = set()

        # leaf_nodes_d = [l for l in get_nodes_by_type(spn, Parametric) if l.scope[0] == d]
        nodes_d = node.children
        print('D', d, nodes_d)
        # clrs = sns.color_palette(cmap, n_colors=len(nodes_d) + 1)
        # act_clrs = []
        part_used = set([part_map[n] for n in nodes_d if leaf_weights[n] >= threshold])
        clrs = sns.color_palette(cmap, n_colors=len(part_used) + 1)

        # act_clrs = []
        act_clrs = {p: c for p, c in zip(part_used, clrs)}
        act_clrs_list = []
        for j, l in enumerate(nodes_d):

            w = leaf_weights[l]

            if w >= threshold:
                # if l.scope[0] == d:

                pdf_x, pdf_y = approximate_density_d(l, X, d, bins=bins, meta_type=meta_types[d])
                pdf_y = pdf_y * w

                mode_id = np.argmax(pdf_y)
                mode_x = pdf_x[mode_id]
                mode_y = pdf_y[mode_id]

                part_id = part_map[l]
                ax.plot(pdf_x, pdf_y, label="{} (part # {})".format(l.__class__.__name__,
                                                                    # part_id), color=clrs[j])
                                                                    part_id), color=act_clrs[part_id])
                ax.text(mode_x, mode_y + mode_y * TEXT_Y_POS_PERC, '{}'.format(part_id))
                # act_clrs.append(clrs[j])
                act_clrs_list.append(act_clrs[part_id])

                x = X[l.row_ids, d]
            # print('LEN x', f_id, l.name, len(l.row_ids))
                miss_vals = np.isnan(x)
                x = x[~miss_vals]
                m_data.append(x)
                all_ids.update(np.array(l.row_ids)[~miss_vals])
        rem_ids = list(sorted(set(np.arange(N)) - all_ids))
        x = X[rem_ids, d]
        miss_vals = np.isnan(x)
        x = x[~miss_vals]
        m_data.append(x)
        # act_clrs.append(clrs[-1])
        act_clrs_list.append(clrs[-1])
        ax.hist(m_data,
                # bins,
                histtype='bar',
                density=True,
                # density=False,
                # stacked=True,
                stacked=False,
                # histtype='bar',
                color=act_clrs_list, alpha=.5)

        ax.legend()
        plt.title('Feature {}'.format(d))

        if show_fig:
            plt.show()

        if save_fig:
            output_path = os.path.join(save_fig, 'd{}-fit'.format(d))
            pp = PdfPages(output_path)
            pp.savefig(fig)
            pp.close()

        plt.close()
