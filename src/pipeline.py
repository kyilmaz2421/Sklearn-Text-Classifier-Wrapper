from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

class pipeline:
    def __init__(self,data_set,model):
        self.model = model

        if data_set == 1:
            self.train = pd.read_csv('20_newsgroup_train.csv',na_values=["NaN"," "]).dropna(axis="rows")
            self.test = pd.read_csv('20_newsgroup_test.csv',na_values=["NaN"," "]).dropna(axis="rows")

        self.train["text"] = self.train["text"].replace("[,'#_<>^-]","",regex=True)
        self.test["text"] = self.test["text"].replace("[,'#_<>^-]","",regex=True)
        
        self.train = self.train.drop('Unnamed: 0',axis=1)
        self.test = self.test.drop('Unnamed: 0',axis=1)
        
        self.train = self.train.sample(frac=1).reset_index(drop=True)
        self.test = self.test.sample(frac=1).reset_index(drop=True)

        self.Y_train = self.train["target"].to_numpy()
        self.Y_test = self.test["target"].to_numpy()

        self.X_train = None
        self.X_test = None

    
    def pre_process(self,stop_words,min_df,max_features,ngram_upper_bound):
        train_size = len(self.train["text"])
        test_size = len(self.test["text"])
        vertical_stack = pd.concat([self.train, self.test], axis=0)
        tfidf_transformer = TfidfTransformer()
        if max_features > 0:
            vectorizer = CountVectorizer(ngram_range=(1, ngram_upper_bound),min_df = min_df, max_features= max_features,stop_words=stop_words)
        else:
            vectorizer = CountVectorizer(ngram_range=(1, ngram_upper_bound),min_df = min_df,stop_words=stop_words)
        
        formated_data = vectorizer.fit_transform(vertical_stack["text"])
        formated_data = tfidf_transformer.fit_transform(formated_data).toarray()
        self.X_train = formated_data[0:train_size][:]
        self.X_test = formated_data[train_size:][:]


    def fit(self):
        if self.model == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(random_state=0).fit(self.X_train,self.Y_train)
            predicted = clf.predict(self.X_test)
            print(predicted)
            print(np.mean(predicted == test["target"].to_numpy()))
       




if __name__ == "__main__":
    p = pipeline(1,"LogisticRegression")
    p.pre_process([],0,0,2)
    print("fit")
    #p.fit()