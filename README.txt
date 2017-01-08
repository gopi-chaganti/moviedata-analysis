Input Data for First part:

Movie database consisting of 58788 movies with 24 different parameters.
The data set contains the following fields:
•	title. Title of the movie.
•	year. Year of release.
•	budget. Total budget (if known) in US dollars
•	length. Length in minutes.
•	rating. Average IMDB user rating.
•	votes. Number of IMDB users who rated this movie.
•	r1-10. Multiplying by ten gives percentile (to nearest 10%) of users who rated this movie a 1.
•	mpaa. MPAA rating.
•	action, animation, comedy, drama, documentary, romance, short. Binary variables representing if movie was classified as belonging to that genre.


Data decimation:
----------------
Random function has been used to do random sampling
K -means clustering has been used to do adaptive sampling

Sampled 1500 rows out of 58788.

Right after starting the server sampling will be done and sampled data will be stored in the memory for later operations.
I have stored the results, as running K-means for every request will take lot of time and make the user wait for quit a noticeable time.

Dimensionality reduction:

Used PCA, MDS(cosine, Euclidean, correlation), ISOMAP for performing dimensionality reduction.

After performing PCA selected first two dimensions in the decreasing order of their Eigen values as it represents 98% of variance in the data

For every request from client selected dimensionality reduction method will be performed on the pre-sampled data.


Visualization:
--------------
Final Data has been visualized using d3.js scatter plots.

If user choose adaptive sampling then data points from different clusters have been shown in different colors.

Constructed a hover card for each data point in the final plot showing it’s title and corresponding coordinates

Scree plot for the sampled data was shown on the left side of the screen.

Observations:

•	The movies I know have surprisingly comes as outlies in each of the plot (This is completely justified as I only watch famous and good movies)

•	This these movies were significantly separated from the rest of the unknown movies.

•	We can apply log transformation to properly visualize the unknown (non significant) movies. Because these results were skewed by outliers (good movies with good ratings).

•	MDS correlation and cosine are giving almost similar kind of results.

•	If I use random sampling most of the good moves were missing in the final plot. Because the ratio of good movies in the total dataset is very less when compared to original dataset, So probability of these movies getting sampled is very low.
•	But If I use adaption sampling then significant portion of these movies will get sampled



Text Analysis:
---------------
I have taken 20,000 bloggers information (http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm) 
each consisting of 10-500 posts

I have parsed these documents and stored the results in a list, in which which each element stores all of one blogger.

Then I have passed this dataset to TfidfVectorizer which stems words and then removes stop words and finally constructs a matrix of documents and features (selected terms)

Using TruncatedSVD LSA analysis has been performed on the data obtained from previous operation. This will give us a matrix of documents and concepts

Finally this result will be passed to k-means clustering algorithm to find the most frequent patterns in each cluster

For each request from user this entire process will be performed, so results will slightly vary for each reaload

Final plot shows the frequent terms for each clusters.
Size of circle shown in the plot represents the cluster size.


Interesting Observations:

Results have been surprising to me. LSA conceptualized the features very well and final clustering results have been very good.
For example: one cluster contains words related to US politics and another about internet










	

