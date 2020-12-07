from time import time
import csv
from pprint import pprint

import pandas as pd
import numpy as np
import nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


def lematize_data(data):
    nltk.download('wordnet')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return data.apply(
        lambda sentence: " ".join([lemmatizer.lemmatize(word) for word in sentence.split(" ")]))


def stem_data(data):
    stemmer = nltk.stem.snowball.SnowballStemmer("english")
    return data.apply(
        lambda sentence: " ".join([stemmer.stem(word) for word in sentence.split(" ")]))


class Classifier:
    def __init__(self,
                 model,
                 stem=False,
                 lematize=False,
                 test_size=0.1,
                 csv_file=None,
                 data=None):
        if csv_file != None and data == None:
            data = pd.read_csv(csv_file, header=None,
                               delimiter=",").sample(frac=1).sample(frac=1)

        if stem:
            data[0] = stem_data(data[0])
        if lematize:
            data[0] = lematize_data(data[0])

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            data[0].to_numpy(), data[1].to_numpy(), test_size=test_size, random_state=42)
        self.model = model
        self.clf = None  # will be updated by best result in grid_search
        self.score_dict = None
        self.param_occurence = None

    def get_nltk_stop_words(self):
        nltk.download('stopwords')
        return nltk.corpus.stopwords.words('english')

    def fit(self, parameters, cv=5):  # default 5 paramter for K cross validation

        pipeline = Pipeline(steps=[
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', self.model),
        ])

        self.clf = GridSearchCV(pipeline,
                                parameters,
                                cv=cv,
                                n_jobs=-1,
                                verbose=5,
                                refit=True,
                                return_train_score=True)

        stop_words_title = {}
        if parameters.get('vect__stop_words'):
            temp = parameters.get('vect__stop_words')
            for i in range(len(temp)):
                if temp[i] == None:
                    stop_words_title[temp[i]] = i
                else:
                    stop_words_title[len(temp[i])] = i

        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        t0 = time()
        self.clf.fit(self.X_train, self.Y_train)
        print("done in %0.3fs" % (time() - t0))
        print()

        print("scores!")
        means = self.clf.cv_results_['mean_test_score']
        stds = self.clf.cv_results_['std_test_score']
        params = self.clf.cv_results_['params']
        self.score_dict = {}
        i = 0
        for mean, std, param in zip(means, stds, params):
            if param.get('vect__stop_words'):
                param['vect__stop_words'] = stop_words_title[len(param['vect__stop_words'])]

            print("mean: %0.3f std: (+/-%0.03f) for %r" % (mean, std * 2, param))
            i += 1
            self.score_dict[(mean, i)] = param  # the i exists to avoid collisions

        print("Best score:")
        print("%0.3f (+/-%0.03f)" % (self.clf.best_score_, std * 2))
        print("with parameters set:")
        best_parameters = self.clf.best_estimator_.get_params()
        if best_parameters.get('vect__stop_words'):
            best_parameters['vect__stop_words'] = stop_words_title[len(
                best_parameters['vect__stop_words'])]
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

    def eval_best_n_params(self, n):
        if n >= 1 or n <= 0:
            n = 0.2
        scores = sorted(self.score_dict.keys(), key=lambda tup: tup[0])
        scores = scores[int(n * len(scores)):]
        p = self.score_dict[scores[0]]
        self.param_occurence = [
        ]  # will be an array of dicts -> each index represnt a param (i.e alpha)
        j = 0
        print("Finding most common params for the top " + str(len(scores)) + " values")
        for k in p.keys():
            # each k is a key to a the specific hyper-param and (i.e k:alpha)
            # we then iterate through our top params dictionary to update occurence of certain param
            self.param_occurence.append({})
            for s in scores:
                # for the given score s gives us the hyper param type giving us the selected value (i.e s -> alpha -> 0.001)
                val = self.score_dict[s][k]
                if self.param_occurence[j].get(val):
                    self.param_occurence[j][val] += 1
                else:
                    self.param_occurence[j][val] = 1
            j += 1

        print(self.param_occurence)

    def generate_predict_set(self, predict_set):
        x_test, y_test = None, None
        if predict_set:
            return np.array(predict_set[0]), np.array(predict_set[1])
        else:
            return self.X_test, self.Y_test

    def prediction(self,
                   include_values=True,
                   predict_set=None,
                   evaluate_prediction=True):
        print("Evaluation on test set:\n")
        x_test, y_test = self.generate_predict_set(predict_set)
        res = self.clf.predict(x_test)
        probabilities = self.clf.predict_proba(x_test)
        if evaluate_prediction:
            print('Accuracy Score : ' + str(accuracy_score(y_test, res)))
            print('Precision Score : ' + str(precision_score(y_test, res, average='micro')))
            print('Recall Score : ' + str(recall_score(y_test, res, average='micro')))
            print('F1 Score : ' + str(f1_score(y_test, res, average='micro')))

            # confusion matrix
            self.plot_cm(title_options, include_values)

        return res, probabilities

        # runs on test set so only use at the end
    def plot_cm(self,
                title_options=[("Confusion Matrix", None)],
                include_values=False,
                predict_set=None):
        x_test, y_test = self.generate_predict_set(predict_set)

        # title_options is a list of tuples with the parametes so we can see multiple matricies
        for title, normalize in title_options:
            disp = plot_confusion_matrix(estimator=self.clf,
                                         X=x_test,
                                         y_true=y_test,
                                         normalize=normalize,
                                         cmap=plt.cm.Blues,
                                         include_values=include_values)
            disp.ax_.set_title(title)
            plt.xticks(rotation=90)
            print(title)

        plt.show()

    def learning_curve(self, train_sizes):
        if train_sizes == []:
            train_sizes = [0.33, 0.66, 1.0]

        plt.figure()
        plt.title("title")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(self.clf.best_estimator_,
                                                                self.X_train,
                                                                self.Y_train,
                                                                cv=5,
                                                                n_jobs=-1,
                                                                train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes,
                         train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std,
                         alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes,
                         test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std,
                         alpha=0.1,
                         color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        plt.legend(loc="best")
        plt.show()
