# Image matching can be done using critical nets (Gu et al. 2010).
# Local features are connected into ‘constellations’.
# identification of which features are invariant within clutches
# find matching features
# identify what kind of feature
# and subsequently whether these features are prioritised in egg
# rejections.
# with these features, rejection success rate up
import pickle
import numpy as np
import pandas as pd
from criticalnet.criticalnet import draw_net, nx
from criticalnet.criticalnet.data.sample import tanmay_images
import cv2
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn import metrics
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import statistics

def edge_difference(e_1, e_2):
    v_1 = (e_1[1][0] - e_1[0][0], e_1[1][1] - e_1[0][1])
    v_2 = (e_2[1][0] - e_2[0][0], e_2[1][1] - e_2[0][1])
    return np.sqrt((v_1[0] - v_2[0]) ** 2 + (v_1[1] - v_2[1]) ** 2)

pickle_in = open("full_graph.pickle", "rb")
graphs = pickle.load(pickle_in)
graphs_length = len(graphs)

images, image_name_list = tanmay_images()

# match_edges_matrix = [[[] for x in range(graphs_length)] for y in range(graphs_length)]
# num_match_edges_matrix = [[0 for x in range(graphs_length)] for y in range(graphs_length)]
pickle_matrix_in = open("full_matrix.pickle", "rb")
match_edges_matrix, num_match_edges_matrix = pickle.load(pickle_matrix_in)

def get_matched_features():
    for i, G_f in enumerate(graphs):
        if G_f is None:
            E_f = []
        else:
            E_f = G_f.edges()
        for j in range(i + 1, graphs_length):
            if graphs[j] is None:
                E_g = []
            else:
                E_g = graphs[j].edges()
            match_edges = []
            for e_1 in E_f:
                min1 = 2 ** 10000
                min2 = 2 ** 10000
                match = e_1
                for e_2 in E_g:
                    diff = edge_difference(e_1, e_2)
                    if diff < min1:
                        min1 = diff
                        match = e_2
                    if diff < min2 and diff != min1:
                        min2 = diff
                min_tmp = min1 if min1 != 0 else 0.0001
                if min2 / min_tmp > 2:
                    match_edges.append((e_1, match))
            match_edges_matrix[i][j] = match_edges
            num_match_edges_matrix[i][j] = len(match_edges) / len(E_f) if len(E_f) != 0 else 0
        num_match_edges_matrix[i][i] = 1
        print('%d graph finished' % i)
    a = np.asarray(num_match_edges_matrix)
    i_lower = np.tril_indices(graphs_length, -1)
    a[i_lower] = a.T[i_lower]  # make the matrix symmetric
    np.savetxt('full_num_match_edges_matrix.csv', a, delimiter=',', fmt='%0.2f')
    pickle_out = open("full_matrix.pickle", "wb")
    pickle.dump([match_edges_matrix, num_match_edges_matrix], pickle_out)
    pickle_out.close()

# get_matched_features()

def get_matched_images():
    for i, img_f in enumerate(images):
        graph_f = graphs[i]
        for j in range(i + 1, len(images)):
            graph_g = graphs[j]
            img_g = images[j]
            width = img_f.shape[1]
            length = img_f.shape[0]
            if img_f.shape[1] - img_g.shape[1] < 5 and img_f.shape[0] - img_g.shape[0] < 5 and length > width:
                # print('image shape is the same')
                img_g = cv2.resize(img_g, (width, length), interpolation=cv2.INTER_AREA)
                horizontally_combined_img = np.concatenate((img_f, img_g), axis=1)
                cv2.imwrite('out.png', horizontally_combined_img)
                G = nx.DiGraph()
                match_edges = match_edges_matrix[i][j]
                for edges in match_edges:
                    n_f = edges[0]
                    n_g = edges[1]
                    n_f_1 = n_f[0]
                    n_f_2 = n_f[1]
                    n_g_1 = n_g[0]
                    n_g_2 = n_g[1]
                    new_n_g_1 = (n_g_1[0], n_g_1[1] + width)
                    new_n_g_2 = (n_g_2[0], n_g_2[1] + width)
                    G.add_edge(n_f_1, new_n_g_1)
                    G.add_edge(n_f_2, new_n_g_2)
                kwargs = {'edge_color': 'red'}
                draw_net(G, image=horizontally_combined_img, ax=None, **kwargs)
                old_G = nx.DiGraph()
                if graph_f is not None:
                    old_G.add_edges_from(graph_f.edges())
                if graph_g is not None:
                    graph_g_edges = graph_g.edges()
                for g_edge in graph_g_edges:
                    g_edge_1 = g_edge[0]
                    g_edge_2 = g_edge[1]
                    new_g_edge_1 = (g_edge_1[0], g_edge_1[1] + width)
                    new_g_edge_2 = (g_edge_2[0], g_edge_2[1] + width)
                    old_G.add_edge(new_g_edge_1, new_g_edge_2)
                draw_net(old_G, image=horizontally_combined_img, ax=None)

# get_matched_images()

def rotational_invariance():
    for i, G_f in enumerate(graphs):
        image_f = images[i]
        image_f_width = image_f.shape[1]
        image_f_length = image_f.shape[0]
        if G_f is None:
            E_f = []
        else:
            E_f = G_f.edges()
        for j in range(i + 1, graphs_length):
            image_g = images[j]
            image_g_width = image_g.shape[1]
            image_g_length = image_g.shape[0]
            if (image_f_length > image_f_width and image_g_width > image_g_length) or \
                    (image_f_length < image_f_width and image_g_width < image_g_length):
                if graphs[j] is None:
                    E_g = []
                else:
                    E_g = graphs[j].edges()
                match_edges = []
                for e_1 in E_f:
                    min1 = 2 ** 10000
                    min2 = 2 ** 10000
                    match = e_1
                    for e_2 in E_g:
                        diff = edge_difference(e_1, e_2)
                        if diff < min1:
                            min1 = diff
                            match = e_2
                        if diff < min2 and diff != min1:
                            min2 = diff
                    min_tmp = min1 if min1 != 0 else 0.0001
                    if min2 / min_tmp > 2:
                        match_edges.append((e_1, match))
                match_edges_matrix[i][j] = match_edges
                num_match_edges_matrix[i][j] = len(match_edges) / len(E_f) if len(E_f) != 0 else 0
            else:
                num_match_edges_matrix[i][j] = -1
        num_match_edges_matrix[i][i] = -1
    a = np.asarray(num_match_edges_matrix)
    i_lower = np.tril_indices(graphs_length, -1)
    a[i_lower] = a.T[i_lower]  # make the matrix symmetric
    # invert the pairwise similarity
    distance = 1 - a
    # run mds, in this case with just 2 dimensions
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                       dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(distance).embedding_
    clf = PCA(n_components=2)
    pos = clf.fit_transform(pos)
    fig = plt.figure(1)
    clusters = [name.split("_")[1] for name in image_name_list]
    # plot mds
    # color_dict = {'PS089': 'red', 'PSK09': 'green', 'PST001': 'blue'}
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(clusters)))
    plt.scatter(pos[:, 0], pos[:, 1], alpha=0.8, s=100, lw=0, color=colors)
    plt.show()
    ax = sns.heatmap(distance, cmap="YlGnBu", square=True)
    plt.show()
    # For each clutch we were then able to compute the centroid,
    # which is simply given by the vectorial mean of all the points
    # corresponding to all the eggs of a particular clutch in the MDS subspace.
    current_cluster = clusters[0]
    x_sum = 0
    y_sum = 0
    means = []
    total = 0
    totals = []
    for i, cluster in enumerate(clusters):
        if current_cluster != cluster:
            current_cluster = cluster
            means.append([x_sum / total, y_sum / total])
            totals.append(total)
            x_sum = pos[i][0]
            y_sum = pos[i][1]
            total = 1
        else:
            x_sum += pos[i][0]
            y_sum += pos[i][1]
            total += 1
    means.append([x_sum / total, y_sum / total])
    totals.append(total)
    # We then computed the distances from that centroid over
    # all the member points of that clutch.
    count = 0
    dist = []
    means = np.array(means)
    for i in range(len(means)):
        groupNum = totals[i]
        for j in range(count, count + groupNum):
            X = np.array([pos[j], means[i]])
            dist.extend(pdist(X))
        count = count + groupNum
    # Intraclutch variation was quantified as the mean distance between
    # the elements of the clutch and its centroid,
    # averaged over all clutches for a particular species.
    count = 0
    mean_dist = []
    for i in range(len(totals)):
        current_dist = dist[count:count + totals[i]]
        mean_dist.append(statistics.mean(current_dist))
        count = count + totals[i]
    intra = statistics.mean(mean_dist)
    print("Intra: %f" % intra)
    # Interclutch variation was quantified as the mean of the distances between
    # all the centroids of all the clutches for a particular species.
    centroid_dist = pdist(means)
    inter = statistics.mean(centroid_dist)
    print("Inter: %f" % inter)
    inter_sim_matrix = squareform(centroid_dist)
    ax = sns.heatmap(inter_sim_matrix, cmap="YlGnBu", square=True)
    plt.show()


rotational_invariance()
