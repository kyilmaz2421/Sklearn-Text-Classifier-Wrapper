from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix
from sklearn.model_selection import GridSearchCV,learning_curve
import pandas as pd
import numpy as np
from pprint import pprint
from time import time
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

def plot_confusion_matrix(cm, classes,

                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Code copied off the sklearn plot confusion matrix page
    https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
    """
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class Classifier:
    def __init__(self,data_set,model):
        if data_set == 1: 
            from sklearn.datasets import fetch_20newsgroups
            train = fetch_20newsgroups(subset='train')
            print(type(train))
           
            test = fetch_20newsgroups(subset='test')
            
            ###### ANY EXTRA PRE_PROCESSING FOR NEWS GROUP HERE ( FYI: NEED TO CONVERT TRAIN/TEST TO PANDAS FIRST TO EDIT) ######
             #pd.read_csv(train).dropna(axis="rows").drop('Unnamed: 0',axis=1).sample(frac=1).reset_index(drop=True)
             #pd.read_csv(test).dropna(axis="rows").drop('Unnamed: 0',axis=1).sample(frac=1).reset_index(drop=True)
            
            self.X_train = train.data
            self.X_test = test.data
            self.Y_train = train.target
            self.Y_test = test.target

        else:
            train = pd.read_csv("../data/imbd/train.txt",header=None).astype(str).sample(frac=1).reset_index(drop=True)
            test = pd.read_csv("../data/imbd/test.txt",header=None).astype(str).sample(frac=1).reset_index(drop=True)

            ###### ANY EXTRA PRE_PROCESSING FOR IMBD HERE ######

            self.X_train = train[0]
            self.X_test = test[0]
            self.Y_train = train[1]
            self.Y_test = test[1]

        self.model = model
        self.clf = None

    def fit(self,parameters,cv):
        
        pipeline = Pipeline(steps = [
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', self.model),
        ])

        grid_search = GridSearchCV(pipeline, parameters,cv=5, n_jobs=-1, verbose=5, refit = True,return_train_score = True)

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

    def baseline_fit(self):
        param = {
            'vect__min_df': ([0]),
            'tfidf__use_idf': ([False]),
        }
        # additionally by the default settings
        # max_feature = None
        # ngram_range = (1,1)
        # max_df=1.0
        # lowercase=True
        # default C
        self.fit(param,5)
        self.eval_on_test()

   
    def eval_on_test(self):
        print()
        print("Evaluation on test set:")
        print()
        res = self.clf.predict(self.X_test)
        #cnf_matrix = confusion_matrix(y_test,y_pred)
        print('Accuracy Score : ' + str(accuracy_score(self.Y_test,res)))
        print('Precision Score : ' + str(precision_score(self.Y_test,res, average='micro')))
        print('Recall Score : ' + str(recall_score(self.Y_test,res, average='micro')))
        print('F1 Score : ' + str(f1_score(self.Y_test,res, average='micro')))
        #confusion matrix
        print('Confusion Matrix : \n' + str(confusion_matrix(self.Y_test,res)))
        
        """
        import itertools
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                            title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                            title='Normalized confusion matrix')

        plt.show()
        """


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
                
if __name__ == "__main__":
    from sklearn.ensemble import AdaBoostClassifier
    c = Classifier(0,AdaBoostClassifier())
    c.baseline_fit()

    c.learning_curve([0.33,0.66,1.0])