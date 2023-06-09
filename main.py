import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns






# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():

    datasetPath = os.path.join(os.getcwd(),'wine-clustering.csv')
    dataset = pd.read_csv(datasetPath)


    possibleClusters = range(2, 8)
    fits = []
    score = []

    for k in possibleClusters:
        # train the model for current value of k on training data
        model = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(dataset)

        # append the model to fits
        fits.append(model)

        # Append the silhouette score to scores
        score.append(silhouette_score(dataset, model.labels_, metric='euclidean'))

    plot = sns.lineplot(x=possibleClusters, y=score)
    plt.show()

    print()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
