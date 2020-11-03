import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_score


df = pd.read_excel (r'default of credit card clients.xls')
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)



new_header = df.iloc[0] #grab the first row for the header
df.columns= new_header
df.columns
df = df[1:] #take the data less the header row
df



y=df['SEX']
del df['AGE'] # Age is not an important predictor and is possibly gender biased
del df['SEX']
del df['BILL_AMT2']
del df['BILL_AMT3']
del df['BILL_AMT4']
del df['BILL_AMT5']
del df['BILL_AMT6']


#y=df['default payment next month']


del df['default payment next month']
x=df


#Data Split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 0)
x=xTrain
y=yTrain

rus= RandomUnderSampler(random_state=20)

x_res, y_res =rus.fit_resample(xTrain,yTrain.astype('int'))




def cross_Validation(X,Y,estimator):
    Folds=[]
    y_Folds=[]
    results=[]
    f1=[]
    acc=[]
    roc=[]
    recall=[]
    '''Creating the 8 folds through train test splits'''
    split_11, split_12, y_11,y_12 = train_test_split(X,Y, test_size=.5, random_state=0)
    split_21, split_22, y_21,y_22 = train_test_split(split_11,y_11, test_size=.5, random_state=0)
    split_23, split_24, y_23,y_24 = train_test_split(split_12,y_12, test_size=.5, random_state=0)
    split_23, split_24, y_23,y_24 = train_test_split(split_12,y_12, test_size=.5, random_state=0)
    fold1, fold2, y_fold1,y_fold2 = train_test_split(split_21,y_21, test_size=.5, random_state=0)
    fold3, fold4, y_fold3,y_fold4 = train_test_split(split_22,y_22, test_size=.5, random_state=0)
    fold5, fold6, y_fold5,y_fold6 = train_test_split(split_23,y_23, test_size=.5, random_state=0)
    fold7, fold8, y_fold7,y_fold8 = train_test_split(split_24,y_24, test_size=.5, random_state=0)
    
    Folds.append(fold1)
    y_Folds.append(y_fold1)
    Folds.append(fold2)
    y_Folds.append(y_fold2)
    Folds.append(fold3)
    y_Folds.append(y_fold3)
    Folds.append(fold4)
    y_Folds.append(y_fold4)
    Folds.append(fold5)
    y_Folds.append(y_fold5)
    Folds.append(fold6)
    y_Folds.append(y_fold6)
    Folds.append(fold7)
    y_Folds.append(y_fold7)
    Folds.append(fold8)
    y_Folds.append(y_fold8)
    
    
    for i in range(0,len(Folds)):
        test_set=Folds[i]
        y_test= y_Folds[i]
        del Folds[i]
        del y_Folds[i]
        train_set= pd.concat(Folds)
        y_train= pd.concat(y_Folds)
        
        
        rus = RandomUnderSampler(random_state=0)
        x_resampled, y_resampled = rus.fit_resample(train_set, y_train.astype('int'))
        experiment=estimator.fit(x_resampled,y_resampled.astype('int'))
        acc.append(accuracy_score(yTest.astype('int'),experiment.predict(xTest)))
        recall.append(recall_score(yTest.astype('int'),experiment.predict(xTest)))
        roc.append(roc_auc_score(yTest.astype('int'),experiment.predict(xTest)))
        f1.append(f1_score(yTest.astype('int'),experiment.predict(xTest)))

        Folds.insert(i,test_set)
        y_Folds.insert(i,y_test)
    
    accuracy=sum(acc)/len(acc)
    results.append(accuracy)
    rec=sum(recall)/len(recall)
    results.append(rec)
    auc=sum(roc)/len(roc)    
    results.append(auc)
    f_score=sum(f1)/len(f1)
    results.append(f_score)
        
        
    return results

'''SVM= LinearSVC()
'''
'''tree=DecisionTreeClassifier(max_depth=15) #this model shows to be a little bit biased can predidct gender with accuracy of .6
CV= cross_validate(tree,xTrain,yTrain.astype('int'), cv = 10, scoring='accuracy')
print(sum(CV['test_score'])/10)
CV= cross_validate(tree,xTrain,yTrain.astype('int'), cv = 10, scoring='recall')
print(sum(CV['test_score'])/10)
CV= cross_validate(tree,xTrain,yTrain.astype('int'), cv = 10, scoring='roc_auc')
print(sum(CV['test_score'])/10)
CV= cross_validate(tree,xTrain,yTrain.astype('int'), cv = 10, scoring='f1')
print(sum(CV['test_score'])/10)
'''

'''forest= RandomForestClassifier(class_weight="balanced")

'''
'''Average accuracy for CV Logistic is .695791 with avg TP .2492 and roc .533 and f1 .2586
   Average Accuracy for Tree( max_depth=10) is .71366 with avg TP .62636 and roc .680577 and f1 .482114
   Average Accuracy for Tree is .61929 with avg TP .6378906 and roc .62606 and f1 .41689
   Average Accuracy for Tree( max_depth=5) is .73879 with avg TP .60253 and roc .681940 and f1 .496805
   Average Accuracy for Tree( max_depth=15) is .664375 with avg TP .633789 and roc .653229 and f1 .44618
   Average Accuracy for SVM is .463333333  with avg TP .57578215 and roc .504310 and f1 .24419
   Average Accuracy for RF is .728291 recall is .60097 and roc .6818 and f1 .48552
   Average Accuracy for RF(depth=5) is .738666 recall is ..60234 and roc .68898 and f1 .4965
   Average Accuracy for RF(depth=10 is .71171  recall is .622265625 and roc .679167 and f1 .479845
   '''
   
   #Pre Process Data and Create ROC Curve. 
# code from https://medium.com/datadriveninvestor/computing-an-roc-graph-with-python-a3aa20b9a3fb
def runClassifiers(X, TestX, cfTest, CF):

     
    RF= RandomForestClassifier(max_depth=5, class_weight='balanced')
    RF.fit(X,CF.astype('int'))
    predictProbForest= RF.predict(TestX.astype('int'))

#GET ROC DATA
    
    fpr1, tpr1, thresholds1 = roc_curve(cfTest.astype('int'), predictProbForest, pos_label=1)
    roc_auc = auc_score(fpr1, tpr1)
    
#GRAPH DATA
    plt.figure()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Random Forest ROC')
    plt.plot(fpr1, tpr1, color='blue', lw=2, label='Random Forest ROC area = %0.2f)' % roc_auc)
    plt.legend(loc="lower right")
    plt.show()
if __name__=="__main__":
    RF=RandomForestClassifier(max_depth=5,class_weight='balanced')
    RF2=RandomForestClassifier(max_depth=5,class_weight='balanced_subsample')
    