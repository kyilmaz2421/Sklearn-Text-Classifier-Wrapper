# Introduction

This is a text classification library that is a wrapoer on the sklearn libraries. Using any machine learning model from sklearn (i.e NaiveBayes, MLP, SVM, etc) and the inputted data this library is an easy to use abstraction from sklearn that preprocesses the data, trains the model with grid search, and then evaluates it with a score report, confusion matricies, and learning curves.

# How it works
1. Clone down this repository

2. Import the sklearn model you wish to train i.e `from sklearn.linear_model import LogisticRegression`

3. Set the path to a csv file where the data exists and pass that in as a param i.e `clf = Classifier(LogisticRegression(), file = file_path)` OR create a pandas dataframe of the data and pass that in i.e`clf = Classifier(LogisticRegression(), data = df)`

4. Set up the parameters dictionary where the keys are set as `"<sklearn-pipeline-step>__<pipeline-step-attribute>"` where the possible options for `sklearn-pipeline-step` are set to `vect` for CountVectorizer, `tfidf` for TfidfTransformer, or sklearn model as `clf` and the `pipeline-step-attribute` options are the specific attributes for the respective class i.e `stop_words` as list of words. The values of the dictionary are the actual list of values that the attributes will be set to. In the gridsearch process every combination of values will be tested and the best model will be returned

5. Train the model with `clf.fit(params_trial_1, cv=5)` where cv is set as the k value for k cross vailidation

6. After training, run `clf.eval_best_n_params(0.9)`. Based on the inputted percentage N this returns the paramters of the 1-N% most succesful models so that you can see which parameters lead to a better model

7. Plot the confusion matrix (`clf.plot_cm(include_values=True)`) and Learning curve (`clf.learning_curve([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])`) to evaluate on current best model from grid search

8. Once training the model is complete you can evaluate model on test data using `clf.prediction(evaluate_prediction=True)` which returns the predictions and the prediction probabilities respectivley. 
   - NOTE: if the parameter `evaluate_prediction` is set to `False` (default is `True`) then the function will not plot a confusion matrix and return a report on the scores
   - NOTE: you can set the param `predict_set` to a list of Y values and the system will predict on those instead of the test set. 



# USAGE example


    from sklearn.linear_model import LogisticRegression
            
    file_path = "../<path_to_csv_file>"
    df = pd.read_csv(file_path, delimiter="|", header=None).sample(frac=1).sample(frac=1)
    
    clf = Classifier(LogisticRegression(), data = df)
 
    params_trial_1 = { 
            'vect__max_features': [10000, 20000],
            'vect__ngram_range': [(1,1), (1, 2)],
            'vect__stop_words' : [None, clf.get_nltk_stop_words()],
            'vect__max_df':[0.1, 0.2],
            'tfidf__use_idf':[True, False],
            'tfidf__smooth_idf':[True, False],
            'tfidf__norm': ['l2'],
            'clf__max_iter': [100, 200],
            'clf__solver': ['saga', 'sag'],
            'clf__C':[2, 1]        
    }
    
    #### TEST 1 ####
    clf.fit(params_trial_1,cv=2)
    clf.eval_best_n_params(0.9) # gets top 10% parameters
    clf.plot_cm(include_values=True)
    clf.learning_curve([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    
    #### AFTER MULTIPLE TESTS AND HYPER-PARAMETER TUNING ####
    clf.prediction(evaluate_prediction=True)
 
 
# Future Improvements

- Allow for more flexibility in customizing the appearence of the plots for the confusion matrix and learning curve
- Enable different data formats to be inputted into class (i.e input the Train and Test set separately)


