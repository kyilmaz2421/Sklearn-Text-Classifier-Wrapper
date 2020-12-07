# Sklearn-Text-Classifier-wrapper

# Introduction

This is a text classification library that is a wrapoer on the sklearn library. Using any model class from sklearn and the inputted data this libraray is an easy to use abstraction from sklearn that trains your model with grid search and evaluates it using a confusion matricies and learning curves.

# How to use


    from sklearn.linear_model import LogisticRegression
            
    file_path = "/Users/kaan/Desktop/school/comp550/project/fake_news/combined_data_cleaned_just_prop_2.csv"
    df = pd.read_csv(file_path, delimiter="|", header=None).sample(frac=1).sample(frac=1)
    
    clf = Classifier(LogisticRegression(), data = df)
 
    params_trial_1 = { 
            'vect__max_features': [10000],
            'vect__ngram_range': [(1, 2)],
            'vect__stop_words' : [clf.get_nltk_stop_words()],
            'vect__max_df':[0.1],
            'tfidf__use_idf':[True],
            'tfidf__smooth_idf':[False],
            'tfidf__norm': ['l2'],
            'clf__max_iter': [100], #two extremes
            'clf__solver': ['saga'],
            'clf__C':[2]        
    }
    #### TEST 1 ####
    clf.fit(params_trial_1,cv=2)
    clf.eval_best_n_params(0.9) # gets top 10% parameters
    clf.plot_cm([],include_values=True)
    clf.learning_curve([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    clf.prediction()


