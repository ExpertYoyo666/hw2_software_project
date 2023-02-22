import sys

import kmeanssp

import numpy as np
import pandas as pd


def parse_args():
    argv = sys.argv

    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print("An Error Has Occurred")
        exit(1)

    input_file1 = argv[-2]
    input_file2 = argv[-1]

    df1 = pd.read_csv(input_file1, header=None)
    df2 = pd.read_csv(input_file2, header=None)
    df = pd.merge(df1, df2, how="inner", on=0).sort_values(by=0).drop(0, axis=1)

    N = df.shape[0]

    k = argv[1]
    eps = argv[-3]

    if not k.isdigit():
        print("Invalid number of clusters!")
        exit(1)

    k = int(k)

    if not (1 < k < N):
        print("Invalid number of clusters!")
        exit(1)


    
    if len(argv) == 5:
        iter = 300
    else:
        iter = argv[2]
        
        if not iter.isdigit():
            print("Invalid maximum iteration!")
            exit(1)
        
        iter = int(iter)

        if not (1 < iter < 1000):
            print("Invalid maximum iteration!")
            exit(1)


    try:
        eps = float(eps)
    except ValueError:
        print("An Error Has Occurred")
        exit(1)

    return k, iter, eps, df.to_numpy()


def kmeans_pp(k, datapoints):
    np.random.seed(0)
    centers = []
    centers_idx = []
    idx = np.random.randint(low=0, high=datapoints.shape[0], size=(1,))[0]
    centers.append(datapoints[idx])
    centers_idx.append(idx)

    while len(centers) < k:
        distances = []
        for i, datapoint in enumerate(datapoints):
            distances.append(distance(centers, datapoint))

        dist_sum = np.sum(distances)
        probabilities = np.array(distances) / dist_sum
        new_center_idx = np.random.choice(datapoints.shape[0], size=(1,), p=probabilities)[0]
        centers.append(datapoints[new_center_idx])
        centers_idx.append(new_center_idx)

    return centers, centers_idx


def distance(centers, datapoint):
    return np.min(np.sqrt([np.sum(np.power(center - datapoint, 2)) for center in centers]))


if __name__ == '__main__':
    try:
        k, iter, eps, datapoints = parse_args()
        centers, centers_idx = kmeans_pp(k, datapoints)

        centers_list = [list(centers[i]) for i in range(len(centers))]
        datapoints_list = datapoints.tolist()
        final_centers = kmeanssp.fit(datapoints_list, centers_list, iter, eps)

        print(','.join(str(i) for i in centers_idx))
        for center in final_centers:
            print(','.join('{:0.4f}'.format(i) for i in center))

    except Exception as e:
        print("An Error Has Occurred")