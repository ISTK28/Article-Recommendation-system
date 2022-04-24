import pandas as pd
import scipy.sparse as sparse
import numpy as np
import random
import implicit
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def make_train(ratings, pct_test = 0.2):
    test_set = ratings.copy()
    test_set[test_set != 0] = 1
    training_set = ratings.copy()

    nonzero_inds = training_set.nonzero()
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))

    random.seed(0)

    num_samples = int(np.ceil(pct_test*len(nonzero_pairs)))

    samples = random.sample(nonzero_pairs, num_samples)
    content_inds = [index[0] for index in samples] # Get the item row indices
    person_inds = [index[1] for index in samples] # Get the user column indices


    training_set[content_inds, person_inds] = 0
    training_set.eliminate_zeros()
    
    return training_set, test_set, list(set(person_inds))

def recommend(grouped_df,person_id, sparse_person_content, person_vecs, content_vecs, num_contents = 10):
    begin_recommend = time.time()
    
    # Get the interactions scores from the sparse person content matrix
    person_interactions = sparse_person_content[person_id,:].toarray()

    # Add 1 to everything, so that articles with no interaction yet become equal to 1
    person_interactions = person_interactions.reshape(-1) + 1

    # Make articles already interacted zero
    person_interactions[person_interactions > 1] = 0

    # Get dot product of person vector and all content vectors
    rec_vector = person_vecs[person_id,:].dot(content_vecs.T).toarray()

    # Scale this recommendation vector between 0 and 1
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]

    # Content already interacted have their recommendation multiplied by zero
    recommend_vector = person_interactions * rec_vector_scaled

    # Sort the indices of the content into order of best recommendations
    content_idx = np.argsort(recommend_vector)[::-1][:num_contents]

    # Start empty list to store titles and scores
    titles = []
    scores = []

    for idx in content_idx:
        # Append titles and scores to the list
        titles.append(grouped_df.title.loc[grouped_df.content_id == idx].iloc[0])
        scores.append(recommend_vector[idx])

    recommendations = pd.DataFrame({'title': titles, 'score': scores})

    end_recommend = time.time()
    print("Total Time taken for recommendation: ", (end_recommend - begin_recommend))
    return recommendations

def auc_score(predictions, test):
    #fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)


def calc_mean_auc(training_set, altered_persons, predictions, test_set):
    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set

    content_vecs = predictions[1]
    for person in altered_persons: # Iterate through each user that had an item altered
        training_column = training_set[:,person].toarray().reshape(-1) # Get the training set column
        zero_inds = np.where(training_column == 0) # Find where the interaction had not yet occurred

        # Get the predicted values based on our user/item vectors
        person_vec = predictions[0][person,:]
        pred = person_vec.dot(content_vecs).toarray()[0,zero_inds].reshape(-1)

        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[:,person].toarray()[zero_inds,0].reshape(-1)

        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store

    # End users iteration

    return float('%.3f'%np.mean(store_auc))

# Bayesian Personalized Ranking
def bpr(sparse_content_person,sparse_person_content):
    begin_build = time.time()
    
    model_1 = implicit.bpr.BayesianPersonalizedRanking(factors = 20, regularization = 0.10, iterations = 100)
    
    alpha = 15
    data = (sparse_content_person * alpha).astype('double')
    
    model_1.fit(data)
    
    person_vecs = sparse.csr_matrix(model_1.user_factors)
    content_vecs = sparse.csr_matrix(model_1.item_factors)
     
    content_train, content_test, content_persons_altered = make_train(sparse_content_person, pct_test = 0.2)
    
    end_build = time.time()
    print("\nTotal Time taken in building the BPR model: ", (end_build - begin_build))
    
    return(calc_mean_auc(content_train, content_persons_altered, [person_vecs, content_vecs.T], content_test))
    
# Alternating Least Squares
def als(sparse_content_person,sparse_person_content):
    begin_build = time.time()
    
    model_2 = implicit.als.AlternatingLeastSquares(factors = 20, regularization = 0.10, iterations = 100)

    alpha = 15
    data = (sparse_content_person * alpha).astype('double')

    model_2.fit(data)

    person_vecs = sparse.csr_matrix(model_2.user_factors)
    content_vecs = sparse.csr_matrix(model_2.item_factors)

    content_train, content_test, content_persons_altered = make_train(sparse_content_person, pct_test = 0.2)
    
    end_build = time.time()
    print("\nTotal Time taken in building the ALS model: ", (end_build - begin_build))
    
    return(calc_mean_auc(content_train, content_persons_altered, [person_vecs, content_vecs.T], content_test))

# Logistic Matrix Factorization
def lmf(sparse_content_person,sparse_person_content):
    begin_build = time.time()
    
    model_3 = implicit.lmf.LogisticMatrixFactorization(factors = 20, regularization = 0.10, iterations = 100)
    
    alpha = 15
    data = (sparse_content_person * alpha).astype('double')
    
    model_3.fit(data)
    
    person_vecs = sparse.csr_matrix(model_3.user_factors)
    content_vecs = sparse.csr_matrix(model_3.item_factors)
   
    content_train, content_test, content_persons_altered = make_train(sparse_content_person, pct_test = 0.2)
    
    end_build = time.time()
    print("\nTotal Time taken in building the LMF model: ", (end_build - begin_build))
    print("\n\n")
    return(calc_mean_auc(content_train, content_persons_altered, [person_vecs, content_vecs.T], content_test))
 

def main():
    # articles_df = pd.read_csv("Desktop//Proj_RecommendationSystem//Article Recommendation//shared_articles.csv")
    # interactions_df = pd.read_csv("Desktop//Proj_RecommendationSystem//Article Recommendation//users_interactions.csv")

    articles_df = pd.read_csv("F:/ml cit/ml project/shared_articles.csv")
    interactions_df = pd.read_csv("F:/ml cit/ml project/users_interactions.csv")
    

    print("Attributes in shared_articles.csv:", len(articles_df.axes[1]))
    print("\n")
    print("Attributes in user_interactions.csv", len(interactions_df.axes[1]))
    print("\n")

    commonAttr = int(0)
    for attr in articles_df.axes[1]:
    	for attr2 in interactions_df.axes[1]:
    	    if(attr == attr2):
    	        commonAttr = commonAttr + 1
    
    print("Total CommonAtrributes in the two datasets: ", commonAttr)
    print("\n")

    # Bar plot for total numbers of each values in 'EventType' column in sharedArticles.csv
    plt.title("", size = 30)
    sns.countplot(articles_df['eventType'])
    print("Percentage of Each type of value in the dataset")
    print(articles_df['eventType'].value_counts()*100/len(articles_df))
    plt.show()
    print("\n\n")

    articles_df.drop(['authorUserAgent', 'authorRegion', 'authorCountry'], axis = 1, inplace = True)
    interactions_df.drop(['userAgent', 'userRegion', 'userCountry'], axis = 1, inplace = True)


    print("sharedArticles Head: \n", articles_df.head())
    print("\n\n")
    print("sharedArticles 'eventType' column values: \n", articles_df['eventType'].value_counts())
    print("\n\n")

    articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
    articles_df.drop('eventType', axis = 1, inplace = True)

    print("sharedArticles Info: \n", articles_df.info())
    print("\n\n")

    print("userInteractions Info: \n",interactions_df.info())
    print("\n\n")

    df = pd.merge(interactions_df[['contentId', 'personId', 'eventType']], articles_df[['contentId', 'title']], how = 'inner', on ='contentId')

    print("Merged Dataset head: \n", df.head())
    print("\n\n")

    print("Merged Dataset info: \n", df.info())
    print("\n\n")

    print("Total values in the eventTypes columns types of values: \n", df['eventType'].value_counts())
    print("\n\n")

    event_type_strength = {'VIEW' : 1.0, 'LIKE' : 2.0, 'BOOKMARK' : 3.0, 'FOLLOW' : 4.0, 'COMMENT CREATED' : 5.0}

    # Mapping eventType values to event_type_strngth values 
    df['eventStrength'] = df['eventType'].apply(lambda x: event_type_strength[x])
    print(df.head())
    print("\n\n")

    # Correlation matrix Heat map
    # print("Correlation Matrix: \n")
    # plt.subplots(figsize = (10, 6))
    # plt.title('Correlation Matrix', size = 10)
    # sns.heatmap(df.corr(), annot = True, linewidths = 0.5)

    # Bar Plot for total number of each value in 'EventType' column
    plt.title("", size = 30)
    col=["VIEW","LIKE","BOOKMARK","FOLLOW","COMMENTED"]
    ax=sns.countplot(df['eventType'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

    print("Percentage of Each type of value in the dataset")
    print(df['eventType'].value_counts()*100/len(df))
    plt.show()
    print("\n\n")
    

    print("Types of Unique Values for each attribute :\n", df.nunique())
    print("\n\n")
    for i in ['eventType']:
        print("\n")
        print(i, " :")
        print(df[i].value_counts())

    df.drop_duplicates()
    grouped_df = df.groupby(['personId', 'contentId', 'title']).sum().reset_index()

    print("Grouped_Dataframe Sample: ", grouped_df.sample())
    print("\n\n")
    print("Grouped_DataFrame DataTypes", grouped_df.dtypes)
    print("\n\n")

    grouped_df['title'] = grouped_df['title'].astype("category")
    grouped_df['personId'] = grouped_df['personId'].astype("category")
    grouped_df['contentId'] = grouped_df['contentId'].astype("category")
    grouped_df['person_id'] = grouped_df['personId'].cat.codes
    grouped_df['content_id'] = grouped_df['contentId'].cat.codes

    print("Grouped_Dataframe head()", grouped_df.head())
    print("\n\n")
    
    sparse_content_person = sparse.csr_matrix((grouped_df['eventStrength'].astype(float), (grouped_df['content_id'], grouped_df['person_id'])))
    sparse_person_content = sparse.csr_matrix((grouped_df['eventStrength'].astype(float), (grouped_df['person_id'], grouped_df['content_id'])))

    scores = []
    scores.append(bpr(sparse_content_person, sparse_person_content))
    scores.append(als(sparse_content_person, sparse_person_content))
    scores.append(lmf(sparse_content_person, sparse_person_content))
    
    print("Accuracy for Bayesian Personalized Ranking (BPR): ", scores[0]*100, "%")
    print("Accuracy for Alternating Least Squares (ALS): ", scores[1]*100, "%")
    print("Accuracy for Logistic Matrix Factorization (LMF): ", scores[2]*100, "%")
    
    
    #roc curves
    
    
main()
