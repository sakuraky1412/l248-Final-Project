import numpy as np
import criticalnet as cnet
from scipy.misc import face
from PIL import Image
import networkx as nx
import pickle
from criticalnet.data.sample import tanmay_images
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


def get_cn():
    images, image_name_list = tanmay_images()
    # sorted_images = [img for _, img in sorted(zip(image_name_list, images))]
    graphs = []
    for i, im in enumerate(images):
        net0 = cnet.CriticalNet(image=im, ktimes=110, lap_mode='DOG')
        net0.compute(beta=8, extrema_dist=30, draw=False)
        G = net0.G
        graphs.append(G)
        print("%d image finished." % i)
    pickle_out = open("full_graph.pickle", "wb")
    pickle.dump(graphs, pickle_out)
    pickle_out.close()


get_cn()

def draw_cn():
    images, image_name_list = tanmay_images()
    pickle_in = open("graph.pickle", "rb")
    graphs = pickle.load(pickle_in)
    # for im in images:
    net0 = cnet.CriticalNet(image=images[1], ktimes=110, lap_mode='DOG', G=graphs[1])
    net0.draw()
    net0 = cnet.CriticalNet(image=images[5], ktimes=110, lap_mode='DOG', G=graphs[5])
    net0.draw()

# draw_cn()


def draw_heatmap():
    path_to_csv = "edited_num_match_edges_matrix.csv"
    df = pd.read_csv(path_to_csv, index_col=0)
    # plt.imshow(df,cmap='hot',interpolation='nearest')
    # plt.show()
    ax = sns.heatmap(df, cmap="YlGnBu", square=True)
    plt.show()

# draw_heatmap()
