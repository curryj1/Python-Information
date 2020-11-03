import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
#from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
import random
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score

df = pd.read_excel (r'default of credit card clients.xls')
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)



new_header = df.iloc[0] #grab the first row for the header
df.columns= new_header
df.columns
df = df[1:] #take the data less the header row
df



#y=df['SEX']
del df['AGE'] # Age is not an important predictor and is possibly gender biased
del df['SEX']
del df['BILL_AMT2']
del df['BILL_AMT3']
del df['BILL_AMT4']
del df['BILL_AMT5']
del df['BILL_AMT6']


y=df['default payment next month']


del df['default payment next month']
x=df


#Data Split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 0)
x=xTrain
y=yTrain


'''
tree=LogisticRegression(class_weight='balanced') #this model shows to be a little bit biased can predidct gender with accuracy of .6
CV= cross_validate(tree,xTrain,yTrain.astype('int'), cv = 10, scoring='accuracy')
print(sum(CV['test_score'])/10)
CV= cross_validate(tree,xTrain,yTrain.astype('int'), cv = 10, scoring='recall')
print(sum(CV['test_score'])/10)
CV= cross_validate(tree,xTrain,yTrain.astype('int'), cv = 10, scoring='roc_auc')
print(sum(CV['test_score'])/10)
CV= cross_validate(tree,xTrain,yTrain.astype('int'), cv = 10, scoring='f1')
print(sum(CV['test_score'])/10)'''


'''SVM= LogisticRegression(class_weight='balanced')
CV= cross_validate(,xTrain,yTrain.astype('int'),cv=10, scoring='f1')
print(sum(CV['test_score'])/10)'''
'''tree=DecisionTreeClassifier(class_weight='balanced', max_depth=10) #this model shows to be a little bit biased can predidct gender with accuracy of .6
CV= cross_validate(tree,xTrain,yTrain.astype('int'), cv = 10, scoring='roc_auc')
print(sum(CV['test_score'])/10)
'''

    
'''Average accuracy for CV Logistic is .580072 with avg TP .38465938?
   Average Accuracy for Logistic(balanced) .6761 with recall .6155 and roc .7073 and f1 .4581
   Average Accuracy for Tree( max_depth=10) is .6825903 with avg TP .58658835
   Average Accuracy for Tree is .60763340 with avg TP .40134973
   Average Accuracy for Tree( max_depth=5) is .69999021418370472 with avg TP .57457495
   Average Accuracy for Tree( max_depth=15) is .6584622830022163 with avg TP .522605
   Average Accuracy for SVM is .5048803256241  with avg TP .51061456
   Average Accuracy for Tree(max_depth=10) only will bill variable .58 with avg TP .6 and "less gender biased accuracy of about .44 but can always just flip your choices so really .56"
   Average Accuracy for SVM only will bill variable .5055 with avg TP .61 (mostly predicts 1's) and roc .5865  and f1 .0878
   Average Accuracy for RF is .803 balanced and recall is .3227 and roc .7293590 and f1 .4272
   Average Accuracy for RF(depth=5) is .77655 balanced and recall is .5964 and roc .771440 and f1 .5408766
   Average Accuracy for RF(depth=10 is .779 balanced and recall is .5657 and roc .76921 and f1 .53739
   Average Accuracy for RF is .803 balanced_subsample and recall is .325 and roc .728453 and f1 .42989
   Average Accuracy for RF(depth=5) is .774 balanced_subsample and recall is .5945 and roc .772189 and f1 .540199
   Average Accuracy for RF(depth=10) is .781 balanced_subsample and recall is .5733 and roc .76828 and f1 .5388955
   '''
'''
forest= RandomForestClassifier(class_weight="balanced_subsample", max_depth=10)

CV=cross_validate(forest,xTrain,yTrain.astype('int'),cv=10, scoring='f1')
print(sum(CV['test_score'])/10)'''

'''
CV=cross_validate(forest,xTrain,yTrain.astype('int'),cv=10, scoring='roc_auc')
print(sum(CV['test_score'])/10)

'''

def cross_Validation(X,Y,estimator):
    Folds=[]
    y_Folds=[]
    f1=[]
    acc=[]
    roc=[]
    recall=[]
    results=[]
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
        random.seed(25)
        
        
        ros = RandomOverSampler(random_state=0)
        x_resampled, y_resampled = ros.fit_resample(train_set, y_train.astype('int'))
        experiment=estimator.fit(x_resampled, y_resampled.astype('int'))
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


if __name__ == '__main__':
    print('main')
    print(cross_Validation(xTrain,yTrain,LinearSVC()))
    


