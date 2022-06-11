from tkinter import X
import pandas as pd
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
from sklearn.model_selection import KFold




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
    elif type == 'nb':
        clf = MultinomialNB()
    elif type == 'svc':
        clf = LinearSVC(random_state = 42,tol=1e-5)
    elif type == 'rf':
        clf = RandomForestClassifier()
    elif type == 'pca':
        clf = PCA(n_components=3)
    elif type == 'lgb':
        params_sklearn = {
    'learning_rate':0.1,
    'max_bin':150,
    'num_leaves':32,    
    'max_depth':11,
    'reg_alpha':0.1,
    'reg_lambda':0.2,   
    'objective':'multiclass',
    'n_estimators':300,
    #'class_weight':weight
}       
        clf = lgb.LGBMClassifier(**params_sklearn)
    
    
    return clf

if __name__=='__main__':
    # Load files into DataFrames
    X_train = pd.read_csv("./data/X_train.csv",index_col=[0])
    #X_submission = pd.read_csv("./data/X_test.csv")
    #print(X_submission.shape[0])

 


    X_train = X_train.sample(100000)
 
    X_train['RiseFall'].replace('Rise',1,inplace = True)
    X_train['RiseFall'].replace('Fall',-1,inplace = True)
    X_train['RiseFall'].replace('Equal',0,inplace = True)    
    Y_train = X_train['RiseFall']# 获取标签值
    X_train.dropna()

    #X_train_processed = X_train.drop(columns=['Id',  'Summary','Score','ProductId','UserId'])
    #X_train_processed['Text'] = X_train_processed['Text'].values.astype('U')
    #X_submission_processed = X_submission.drop(columns=['Id', 'Summary', 'Score','ProductId','UserId'])



    X_train_processed = X_train.drop(columns=['user_name','user_location','user_description','date','user_created','user_verified','hashtags','tweet_source','RiseFall'])
    X_train_processed['text'] = X_train_processed['text'].values.astype('U')
    #X_submission_processed = X_submission.drop(columns=['Id', 'Summary','Score'])

    #print(len(X_submission_processed))

    #X_submission_processed['Text'] = X_submission_processed['Text'].values.astype('U')
    #print(len(X_submission_processed))

 

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


    # 切分数据集
    x_train,x_test,y_train,y_test= train_test_split(X_train_processed, Y_train,random_state= 42)


    kf = KFold(n_splits=5)

    pipe.fit(x_train,y_train)
    print('end training.....')


    test_result = pipe.predict(x_test)

    print('accuracy=',accuracy_score(y_test,test_result))
    #print(classification_report(y_test, test_result))
    #print('y_test', y_test)
    #print('pred', test_result)

    # %matplotlib inline
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_test, test_result)
    from mlxtend.plotting import plot_confusion_matrix
    fig, ax = plot_confusion_matrix(conf_mat=CM , figsize=(10, 5))
    plt.show()

    # Predict the score using the model
    #print(len(X_submission_processed))
    #X_submission['Score'] = pipe.predict(X_submission_processed)
    # Create the submission file
    #submission = X_submission[['Id', 'Score']]
    #submission.to_csv("./data/submission.csv", index=False)
