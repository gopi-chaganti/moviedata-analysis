from __future__ import absolute_import, division, print_function
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from os import listdir
from os.path import isfile, join
import re
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from flask import Flask
from flask import render_template, send_from_directory
import pandas
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics.pairwise
from sklearn import (decomposition, manifold)


def peform_lsa():

        files = [f for f in listdir('blogs') if isfile(join('blogs', f))]
        dataset = []

        for i,file in enumerate(files):
            with open('blogs/' + file, 'r') as f:
                if(i > 100):
                    break
                read_data = f.read()
                pattern = r'<post>(.*?)</post>'
                posts = re.findall(pattern, read_data, re.DOTALL)
                dataset.append(unicode(",".join(posts), errors='ignore').encode('ascii','ignore'))
                f.closed

        vectorizer = TfidfVectorizer(max_df=0.5, max_features=1000, min_df=2, stop_words='english', use_idf=True)
        X = vectorizer.fit_transform(dataset)

        n_components = 100
        svd = TruncatedSVD(n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(X)
        explained_variance = svd.explained_variance_ratio_.sum()

        n_clusters = 5
        km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1)
        km.fit(X)

        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]

        counter = Counter(km.labels_)
        clusters = defaultdict(list)
        terms = vectorizer.get_feature_names()
        for i in range(n_clusters):
            for ind in order_centroids[i, :10]:
                clusters[i].append(terms[ind].encode('ascii','ignore'))
            clusters[i].append(counter[i])
        return clusters

def AdaptiveSample(data, length):
    number_of_clusters = 20
    kmeans = cluster.KMeans(n_clusters=number_of_clusters)
    kfit = kmeans.fit(data)
    labels = kmeans.labels_
    data["cluster"] = labels
    sample = data[0:0]
    for index in range(number_of_clusters):
        clusterdata = data[labels==index]
        samplestocollect = int((len(clusterdata) /len(data)) * length)
        if samplestocollect == 0:
            #take atleast one sample from each cluster
            samplestocollect = 1

        samplefromcluster = clusterdata.sample(samplestocollect)
        sample = pandas.concat([sample, samplefromcluster])
    return sample

#MDS
def my_mds(data, similarity):
    # types of distance functions: cosine,correlation,euclidean
    mds = manifold.MDS(n_components =2, max_iter=100, n_init=1, dissimilarity="precomputed")
    similarity_matrix = sklearn.metrics.pairwise.pairwise_distances(data, Y=None, metric=similarity, n_jobs=1)
    Y = mds.fit_transform(similarity_matrix)
    return Y

#ISOMAP
def my_isomap(data):
    X_iso = manifold.Isomap(n_neighbors = 10, n_components=2).fit_transform(data)
    return X_iso

#PCA analysis
def my_pca(data):
    global pca_eigen_values
    pca = decomposition.PCA()
    pca.fit(data)
    print(pca.explained_variance_)
    pca_eigen_values = pca.explained_variance_.tolist()
    pca.n_components = 2
    reduced_data = pca.fit_transform(data)
    return reduced_data

def remaptitle(x):
    return title_index_mapping[x]

def remapcluster(x):
    return cluster_index_mapping[x]


#read the data
data = pandas.read_csv('movies.csv', sep=',',low_memory=False)
index_column = data.columns.values[0]
title_index_mapping = dict(zip(data[index_column], data["title"]))

#clean and preprocess the data
del data['mpaa']
del data['title']
del data['budget']

#pick random sample from data
random_sample =  data.sample(100)

#pick adaptive sample
print("adaptive sampling")
adaptive_sample  = AdaptiveSample(data, 100)
cluster_index_mapping = dict(zip(adaptive_sample[adaptive_sample.columns.values[0]], adaptive_sample["cluster"]))
print("done")

pca_eigen_values = []


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/lsa.html")
def lsa_template():
    return render_template("lsa.html")

@app.route("/numerical.html")
def numerical_analysis():
    return render_template("numerical.html")

@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('static/', path)

@app.route("/screeplot")
def screeplot():
    pca_eigen_values.sort(reverse=True)
    return pandas.json.dumps(pandas.DataFrame(pca_eigen_values))

@app.route("/random/pca")
def randomsampling_pca():
    reduced_data = my_pca(random_sample)
    #log transformation
    reduced_data = pandas.DataFrame(reduced_data)
    reduced_data["title"] = list(map(remaptitle, random_sample[index_column]))
    return pandas.json.dumps(reduced_data)

@app.route("/random/mds_cosine")
def randomsampling_mds_consine():
    reduced_data = my_mds(random_sample, "cosine")
    reduced_data = pandas.DataFrame(reduced_data)
    reduced_data["title"] = list(map(remaptitle, random_sample[index_column]))
    return pandas.json.dumps(reduced_data)

@app.route("/random/mds_correlation")
def randomsampling_mds_correlation():
    reduced_data = my_mds(random_sample, "correlation")
    reduced_data = pandas.DataFrame(reduced_data)
    reduced_data["title"] = list(map(remaptitle, random_sample[index_column]))
    return pandas.json.dumps(reduced_data)

@app.route("/random/mds_euclidean")
def randomsampling_mds_euclidean():
    reduced_data = my_mds(random_sample, "euclidean")
    reduced_data = pandas.DataFrame(reduced_data)
    reduced_data["title"] = list(map(remaptitle, random_sample[index_column]))
    return pandas.json.dumps(reduced_data)

@app.route("/random/isomap")
def randomsampling_isomap():
    reduced_data = my_isomap(random_sample)
    reduced_data = pandas.DataFrame(reduced_data)
    reduced_data["title"] = list(map(remaptitle, random_sample[index_column]))
    return pandas.json.dumps(reduced_data)

@app.route("/adaptive/pca")
def adaptive_pca():
    reduced_data = my_pca(adaptive_sample)
    reduced_data = pandas.DataFrame(reduced_data)
    reduced_data["title"] = list(map(remaptitle, adaptive_sample[index_column]))
    reduced_data["cluster"] = list(map(remapcluster, adaptive_sample[index_column]))
    return pandas.json.dumps(reduced_data)

@app.route("/adaptive/mds_consine")
def adaptive_mds_consine():
    reduced_data = my_mds(adaptive_sample, "cosine")
    reduced_data = pandas.DataFrame(reduced_data)
    reduced_data["title"] = list(map(remaptitle, adaptive_sample[index_column]))
    reduced_data["cluster"] = list(map(remapcluster, adaptive_sample[index_column]))
    return pandas.json.dumps(reduced_data)

@app.route("/adaptive/mds_correlation")
def adaptive_mds_correlation():
    reduced_data = my_mds(adaptive_sample, "correlation")
    reduced_data = pandas.DataFrame(reduced_data)
    reduced_data["title"] = list(map(remaptitle, adaptive_sample[index_column]))
    reduced_data["cluster"] = list(map(remapcluster, adaptive_sample[index_column]))
    return pandas.json.dumps(reduced_data)

@app.route("/adaptive/mds_euclidean")
def adaptive_mds_euclidean():
    reduced_data = my_mds(adaptive_sample, "euclidean")
    reduced_data = pandas.DataFrame(reduced_data)
    reduced_data["title"] = list(map(remaptitle, adaptive_sample[index_column]))
    reduced_data["cluster"] = list(map(remapcluster, adaptive_sample[index_column]))
    return pandas.json.dumps(reduced_data)

@app.route("/adaptive/isomap")
def adaptive_isomap():
    reduced_data = my_isomap(adaptive_sample)
    reduced_data = pandas.DataFrame(reduced_data)
    reduced_data["title"] = list(map(remaptitle, adaptive_sample[index_column]))
    reduced_data["cluster"] = list(map(remapcluster, adaptive_sample[index_column]))
    return pandas.json.dumps(reduced_data)

@app.route("/textanalysis")
def lsa_analysis():
    reduced_data = pandas.DataFrame(peform_lsa());
    return pandas.json.dumps(reduced_data)


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=12374,debug=True)