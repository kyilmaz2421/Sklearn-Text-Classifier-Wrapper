from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,plot_confusion_matrix
from sklearn.model_selection import GridSearchCV,learning_curve
import pandas as pd
import numpy as np
from pprint import pprint
from time import time
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


class Classifier:
    def __init__(self,data_set,model):
        if data_set == 1: 
            from sklearn.datasets import fetch_20newsgroups
            
            train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
            test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
            
            ###### ANY EXTRA PRE_PROCESSING FOR NEWS GROUP HERE ######
            
            self.X_train = train.data
            self.X_test = test.data
            self.Y_train = train.target
            self.Y_test = test.target
            self.class_names = test.target_names

        else:
            
            train = pd.read_csv("../data/train.txt",header=None).sample(frac=1).reset_index(drop=True)
            test = pd.read_csv("../data/test.txt",header=None).sample(frac=1).reset_index(drop=True)

            train.iloc[:,-1] = train.iloc[:,-1].astype(str).apply(lambda x: "1" if (x in ["5","6","7","8","9","10"]) else "0")
            test.iloc[:,-1] = test.iloc[:,-1].astype(str).apply(lambda x: "1" if (x in [" 5"," 6"," 7"," 8"," 9"," 10"]) else "0")
            
            ###### ANY EXTRA PRE_PROCESSING FOR IMBD HERE ######

            self.X_train = train[0].to_numpy()
            self.X_test = test[0].to_numpy()
            self.Y_train = train[1].to_numpy()
            self.Y_test = test[1].to_numpy()
            self.class_names = ["negative","positive"]

        self.model = model
        self.clf = None #will be updated by best result in grid_search

    def fit(self, parameters, cv):  #default k paramter for K cross validation

        pipeline = Pipeline(steps = [
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', self.model),
        ])

        grid_search = GridSearchCV(pipeline, parameters,cv=5, n_jobs=-1, verbose=5, refit = True, return_train_score = True)

        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        pprint(parameters)
        t0 = time()
        grid_search.fit(self.X_train, self.Y_train)
        print("done in %0.3fs" % (time() - t0))
        print()
        
        print("scores!")
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
            
        print("Best score:")
        print("%0.3f (+/-%0.03f)" % (grid_search.best_score_, std * 2))
        print("with parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
        self.clf = grid_search
   
    def eval_on_test(self, title_options,include_values):
        print(title_options)
        print("Evaluation on test set:")
        print()
        res = self.clf.predict(self.X_test)
        print('Accuracy Score : ' + str(accuracy_score(self.Y_test,res)))
        print('Precision Score : ' + str(precision_score(self.Y_test,res, average='micro')))
        print('Recall Score : ' + str(recall_score(self.Y_test,res, average='micro')))
        print('F1 Score : ' + str(f1_score(self.Y_test,res, average='micro')))
        #confusion matrix
        if title_options==[]: title_options = [("Confusion Matrix",None)]
        self.plot_cm(title_options,include_values)

    def plot_cm(self,title_options, include_values):
        #produces multiple cnf matricies
        #title_options is a list of tuples with the parametes so we can see multiple matricies
        for title, normalize in title_options:
            disp = plot_confusion_matrix(estimator=self.clf, X=self.X_test, y_true=self.Y_test,normalize=normalize,
                                         display_labels=self.class_names, cmap=plt.cm.Blues, include_values=False)
            disp.ax_.set_title(title)
            plt.xticks(rotation=90)
            print(title)
            print(disp.confusion_matrix)

        plt.show()

    def learning_curve(self,train_sizes):
        #[0.33,0.66,1.0]
        plt.figure()
        plt.title("title")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            self.clf.best_estimator_, self.X_train, self.Y_train, cv=5, n_jobs=-1, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                    label="Cross-validation score")

        plt.legend(loc="best")
        plt.show()
    
    def dummy_fit(self): 
        from sklearn.dummy import DummyClassifier
        print("Using sklearn dummy classifier and predicting on test data to get a baseline worst case score")
        dummy_clf = DummyClassifier()
        print("Training dummy classifier...")
        dummy_clf.fit(self.X_train,self.Y_train)
        print("Results in baseline 'random' classifier:")
        print(dummy_clf.score(self.X_test, self.Y_test))


if __name__ == "__main__":
    pass