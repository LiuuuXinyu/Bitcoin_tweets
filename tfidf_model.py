from tkinter import X
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from nltk.corpus import stopwords
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score




import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)


def make_tfidf(reviews_bow):
    '''
    :param reviews_bow: text list
    :return: X_train
            vectorizer
    '''
    stopword = set(stopwords.words('english')) # get stop words
    vectorizer = tfidf(stop_words= stopword)
    vectorizer.fit_transform(reviews_bow)
    return vectorizer

def model_type(type):
    print('start training.....')
    if type == 'lr':
        clf = LogisticRegressionCV()
    elif type == 'svc':
        clf = LinearSVC(random_state = 42,tol=1e-5)
    elif type == 'rf':
        clf = RandomForestClassifier()
    elif type == 'pca':
        clf = PCA(n_components=3)
    elif type == 'lgb':
        params_sklearn = {
    'learning_rate':0.05,
    'max_bin':150,
    'num_leaves':32,    
    'max_depth':11,
    'reg_alpha':0.1,
    'reg_lambda':0.2,   
    'n_estimators':500,
    #'class_weight':weight
}       
        clf = lgb.LGBMClassifier(**params_sklearn)
    
    
    return clf

if __name__=='__main__':
    # Load files into DataFrames
    X_train = pd.read_csv("./data/X_train.csv",index_col=[0])
    #X_submission = pd.read_csv("./data/X_test.csv")
    #print(X_submission.shape[0])

 


 
    X_train['RiseFall'].replace('Rise',1,inplace = True)
    X_train['RiseFall'].replace('Fall',0,inplace = True)
    X_train['RiseFall'].replace('Equal',1,inplace = True)  
    class_1 = X_train[X_train['RiseFall']==1].sample(n=50000)
    class_0 = X_train[X_train['RiseFall']==0].sample(n=50000)
    class_1 = class_1.append(class_0)
    X_train=class_1
    X_train.dropna()
    Y_train = X_train['RiseFall']



    X_train_processed = X_train.drop(columns=['user_name','user_friends','user_location','user_description','user_favourites','user_followers','date','user_created','user_verified','hashtags','tweet_source','RiseFall'])
    X_train_processed['text'] = X_train_processed['text'].values.astype('U')

    X_train_processed['Helpfulness'].fillna(X_train_processed['Helpfulness'].mean(),inplace = True)
    X_train_processed = X_train_processed.replace((np.inf,-np.inf,np.nan),0)

 

# initialise model and vectorizers
     
    model = model_type('lgb')
    stopword = set(stopwords.words('english'))#non-stopwords has better performance
    vectorizer1 = TfidfVectorizer()


# construct the column transfomer
    column_transformer = ColumnTransformer(
    [('tfidf1', vectorizer1, 'text')],
    remainder='passthrough')


# fit the model
    pipe = Pipeline([
                  ('tfidf', column_transformer),
                  ('classify', model)
                ])


    x_train,x_test,y_train,y_test= train_test_split(X_train_processed, Y_train,random_state= 42)




    pipe.fit(x_train,y_train)
    print('end training.....')

    
    test_result = pipe.predict(x_test)

    print('accuracy=',accuracy_score(y_test,test_result))
    #print(classification_report(y_test, test_result))
    #print('y_test', y_test)
    #print('pred', test_result

    #kfold

    print('kfold .....')

    scores = cross_val_score(pipe,x_train,y_train,cv=10)
    print('test accuracy:%s'%scores)
    print('CV accurecy:%3f +/- %.3f'%(np.mean(scores),np.std(scores)))


    # %matplotlib inline
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_test, test_result)
    from mlxtend.plotting import plot_confusion_matrix
    fig, ax = plot_confusion_matrix(conf_mat=CM , figsize=(10, 5))
    plt.show()


