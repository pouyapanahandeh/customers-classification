# Pouya Panahandeh neptun code => S7RFBI

from numpy.lib.function_base import average
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.arrays import categorical
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from sklearn.linear_model import RidgeClassifier
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import seaborn as sns

# list of functions

# line seperator
def line_seperator():
    print("\n","----------------------------------------------------------------","\n")


# step 3: calculate f measure scores
def f_measure_score(y_true, y_pred): # focus more on recall, that's why we put beta equal to 2
	return fbeta_score(y_true, y_pred, beta=2)


def recall_score(y_true, y_pred):
    return  recall_score(y_true, y_pred, average='binary') 


def evaluate_model_scores(input_data, output_data, model):
	kfold_evaluation_procedure = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	metric = make_scorer(f_measure_score)
	scores = cross_val_score(model, input_data, output_data, scoring=metric, cv=kfold_evaluation_procedure, n_jobs=-1)
	return scores


# step 4: Define model to test and evaluate accuaracy
def get_models():
	models, names = list(), list()
	# LR
	models.append(LogisticRegression(solver='liblinear'))
	names.append('LR')
	# LDA
	models.append(LinearDiscriminantAnalysis())
	names.append('LDA')
	# NB
	models.append(GaussianNB())
	names.append('NB')
	# GPC
	models.append(GaussianProcessClassifier())
	names.append('GPC')
	# SVM
	models.append(SVC(gamma='scale'))
	names.append('SVM')
	return models, names

# step 4+: undersampling model
def get_us_models():
    us_models, us_names = list(), list()
	# TL
    us_models.append(TomekLinks())
    us_names.append('TL')
	# ENN
    us_models.append(EditedNearestNeighbours())
    us_names.append('ENN')
	# RENN
    us_models.append(RepeatedEditedNearestNeighbours())
    us_names.append('RENN')
	# OSS
    us_models.append(OneSidedSelection())
    us_names.append('OSS')
	# NCR
    us_models.append(NeighbourhoodCleaningRule())
    us_names.append('NCR')
    return us_models, us_names

# readingfile 
file_data = 'project_data.csv'
original_data = pd.read_csv(file_data, header=None,skiprows=1)

# number of rows and columns
print("number of row and columns", original_data.shape)

# step 1: dataset: cleaning data
# cleaning data  
orginal_data = original_data.drop_duplicates()  # remove duplicates 
original_data = orginal_data.dropna()           # remove empty cell
print("number of row and columns after cleaning", orginal_data.shape)
line_seperator()

# step 1.1: datset information and data visualization
print("data set information and data visualization: \n")
good_and_bad_customer = original_data.values[:, -1]
print("combination of good and bad customer: ",len(good_and_bad_customer))
print(type(good_and_bad_customer), "shape: ",good_and_bad_customer.shape, good_and_bad_customer.dtype)

# 1 => good customer, 2 => bad customer
count_good_customer = np.count_nonzero(good_and_bad_customer == 1)
count_bad_customer = np.count_nonzero(good_and_bad_customer == 2)
good_percentage = (count_good_customer / len(good_and_bad_customer)) * 100
bad_percentage = (count_bad_customer / len(good_and_bad_customer)) * 100
print("The number of good customer", count_good_customer, good_percentage,"%", "The number of bad customer", count_bad_customer, bad_percentage,"%","\n")
print(original_data.head(), "\n")
line_seperator()

# data visualization
bar_chart = original_data.groupby(good_and_bad_customer).size()
bar_chart.plot(kind='bar')
#plt.show()

# step 2: split data, featuring(hot encoding)
result_column = len(original_data.columns) - 1
input_data = original_data.drop(result_column, axis=1)
output_data = original_data[result_column]

categorical_columns = input_data.select_dtypes(include=['object', 'bool']).columns
numerical_columns = input_data.select_dtypes(include=['int64', 'float64']).columns
categorical_encode = ColumnTransformer([('o',OneHotEncoder(),categorical_columns)], remainder='passthrough')
input_data = categorical_encode.fit_transform(input_data)
output_data = LabelEncoder().fit_transform(output_data)

# step 3: calculate fbeta_score, evaluate model against DummyClassifier
print(input_data.shape, output_data.shape)
# dummyclassifier
model = DummyClassifier(strategy='constant', constant=1)
scores = evaluate_model_scores(input_data, output_data, model)
print('mean F-measure score: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
# print(output_data.shape, type(output_data))
line_seperator()



# step 4: evaluate different model
models, names = get_models()
results = list()

# printing the result of each model 
for i in range(len(models)):
    columns_transformer = ColumnTransformer([('c',OneHotEncoder(),categorical_columns), ('n',MinMaxScaler(),numerical_columns)])
    pipeline = Pipeline(steps=[('t',columns_transformer),('m',models[i])])
    model_scores = evaluate_model_scores(input_data, output_data, pipeline)
    results.append(model_scores)
    print('=>%s %.3f (%.3f)' % (names[i], np.mean(scores), np.std(scores)))   

line_seperator()

# step 4+: evaluate different undersampling model
us_models, us_names = get_us_models()
us_results = list()

# printing the result of each models
for j in range(len(us_models)):
	us_model = LogisticRegression(solver='liblinear', class_weight='balanced')
	# one hot encode categorical, normalize numerical
	ct = ColumnTransformer([('c',OneHotEncoder(),categorical_columns), ('n',MinMaxScaler(),numerical_columns)])
	# scale, then undersample, then fit model
	us_pipeline = Pipeline(steps=[('t',ct), ('s', us_models[j]), ('m',us_model)])
	# evaluate the model and store results
	us_scores = evaluate_model_scores(input_data, output_data, us_pipeline)
	us_results.append(scores)
	# summarize and store
	print('=>%s %.3f (%.3f)' % (us_names[j], np.mean(us_scores), np.std(us_scores)))


line_seperator()

# step 4++: improving results
smtn_model = RidgeClassifier()
sampling = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
cts = ColumnTransformer([('c',OneHotEncoder(),categorical_columns), ('n',StandardScaler(),numerical_columns)])
# scale, then sample, then fit model
pipeline = Pipeline(steps=[('t',cts), ('s', sampling), ('m',smtn_model)])
# evaluate the model and store results
scoress = evaluate_model_scores(input_data, output_data, pipeline)
print('%.3f (%.3f)' % (np.mean(scoress), np.std(scoress)))

line_seperator()


# step 5: Clustering
# find the relationship between each columns
# print(original_data.corr())
x = original_data.iloc[:,[1,4,7,10,12,15,17,20]]
cl_score = []
for cluster in range(1,12):
    kmeans = KMeans(n_clusters = cluster, init="k-means++", random_state=10)
    kmeans.fit(x)
    cl_score.append(kmeans.inertia_)



plt.plot(range(1,12), cl_score)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
# plt.show() uncomment to see the chart

# Silhouette score
for n_clusters in range(2,11):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(x) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(x)
    silhouette_avg = silhouette_score(x, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(x, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        print(2)
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(x.iloc[:, 0], x.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

# plt.show() uncomment to see the cluster chart



# # step 6: test model by making prediction on given data
# Good 0:
# >Predicted=0 (expected 0)
# >Predicted=0 (expected 0)
# >Predicted=1 (expected 0)
# Bad 1:
# >Predicted=1 (expected 1)
# >Predicted=1 (expected 1)
# >Predicted=1 (expected 1)