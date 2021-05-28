# Program for classifying song lyrics by genre.
# Trains model on all songs in dataset, weighting genres equally.

import pandas as pd
from pandas import DataFrame
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sn
from numpy import ndarray

#%matplotlib inline
# Import libraries for text manipulation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
# Import modules for evaluation purposes
# Import libraries for predcton
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,auc,f1_score

# Import WordCloud
from wordcloud import WordCloud

# Read data for each genre into separate pandas dataframes
data_blues = pd.read_csv("tcc_ceds_music_blues.csv", sep=",")
data_country = pd.read_csv("tcc_ceds_music_country.csv", sep=",")
data_hiphop = pd.read_csv("tcc_ceds_music_hiphop.csv", sep=",")
data_jazz = pd.read_csv("tcc_ceds_music_jazz.csv", sep=",")
data_pop = pd.read_csv("tcc_ceds_music_pop.csv", sep=",")
data_reggae = pd.read_csv("tcc_ceds_music_reggae.csv", sep=",")
data_rock = pd.read_csv("tcc_ceds_music_rock.csv", sep=",")

# Concatenate into full dataset
data = pd.concat([data_blues, data_country, data_hiphop, data_jazz, data_pop, data_reggae, data_rock], axis=0)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(data['lyrics'], data['genre'].values, random_state=366, test_size=0.1, shuffle=True)

# Show the size of our datasets
print('X Train Size:',X_train.shape)
print('X Test Size:',X_test.shape)


# Create the numericalizer TFIDF for lowercase
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 1))
# Numericalize the train dataset
train = tfidf.fit_transform(X_train.values.astype('U'))
# Numericalize the test dataset
test = tfidf.transform(X_test.values.astype('U'))

# Show size of numericalized train set
print('Train size: ',train.shape)
# Show size of numericalized test set
print('Test size: ',test.shape)


# Create logistic regression model, train it on the train dataset, and print the scores
# 
model = LogisticRegression()
model.fit(train, y_train)
print("train score:", model.score(train, y_train))
print("test score:", model.score(test, y_test))

# Create, plot confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    ''' Plot the confusion matrix for the target labels and predictions '''
    cm = confusion_matrix(y_test, y_pred)

    # Create a dataframe with the confusion matrix values
    df_cm = pd.DataFrame(cm, range(cm.shape[0]),
                  range(cm.shape[1]))
    #plt.figure(figsize = (10,7))
    # Plot the confusion matrix
    sn.set(font_scale=1.4) #for label size
    sn.heatmap(df_cm, annot=True,fmt='.0f',annot_kws={"size": 10})# font size
    plt.show()

# Predict test set results, print classification report and confusion matrix
y_pred = model.predict(test)
print(metrics.classification_report(y_test, y_pred,  digits=5))
plot_confusion_matrix(y_test, y_pred)

def visualize(genre):
  
  arr = DataFrame.to_numpy(genre)
  corpus = [x[0] for x in arr]
  vecs = tfidf.fit_transform(corpus)
  feature_names = tfidf.get_feature_names()
  dense = vecs.todense()
  lst1 = dense.tolist()
  df = pd.DataFrame(lst1, columns=feature_names)

  wordcloud = WordCloud(height=400, width=400, max_words=50, background_color='white').generate_from_frequencies(df.T.sum(axis=1))

  plt.imshow(wordcloud)
  plt.axis('off')
  plt.show()

def print_topn(vectorizer, clf, class_labels, n):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        topn = np.argsort(clf.coef_[i])[-n:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in topn)))

print_topn(tfidf, model, model.classes_, 15)

# Code to print word clouds
# visualize(data_blues)
# visualize(data_country)
# visualize(data_hiphop)
# visualize(data_jazz)
# visualize(data_pop)
# visualize(data_reggae)
# visualize(data_rock)