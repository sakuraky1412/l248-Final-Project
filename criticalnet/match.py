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
from criticalnet.data.sample import tanmay_images
import cv2


def edge_difference(e_1, e_2):
    v_1 = (e_1[1][0] - e_1[0][0], e_1[1][1] - e_1[0][1])
    v_2 = (e_2[1][0] - e_2[0][0], e_2[1][1] - e_2[0][1])
    return np.sqrt((v_1[0] - v_2[0]) ** 2 + (v_1[1] - v_2[1]) ** 2)

pickle_in = open("../graph.pickle", "rb")
graphs = pickle.load(pickle_in)

graphs_length = len(graphs)
match_edges_matrix = [[[] for x in range(graphs_length)] for y in range(graphs_length)]
num_match_edges_matrix = [[0 for x in range(graphs_length)] for y in range(graphs_length)]

for i, G_f in enumerate(graphs):
    E_f = G_f.edges()
    for j in range(i+1, graphs_length):
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
            min_tmp = min1 if not min1 == 0 else 0.0001
            if min2 / min_tmp > 1.5 and match is not e_1:
                match_edges.append((e_1, match))
        match_edges_matrix[i][j] = match_edges
        num_match_edges_matrix[i][j] = len(match_edges)/len(E_f)
    num_match_edges_matrix[i][i] = 1

a = np.asarray(num_match_edges_matrix)
i_lower = np.tril_indices(graphs_length, -1)
a[i_lower] = a.T[i_lower]  # make the matrix symmetric
np.savetxt('num_match_edges_matrix.csv', a, delimiter=',', fmt='%0.2f')

images, image_name_list = tanmay_images()
for i, img_f in enumerate(images):
    for j in range(i+1, images):
        img_g = images[j]
        if img_f.shape == img_g.shape:
            print('image shape is the same')
            width = img_g.shape[1]
            length = img_g.shape[0]
            horizontally_combined_img = np.concatenate((img_f, img_g), axis=1)
            cv2.imshow('horizontally_combined_img', horizontally_combined_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()