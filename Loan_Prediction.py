import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import grid
from unittest.mock import inplace
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from nltk.chunk.util import accuracy
from sklearn.cross_validation import KFold
from sklearn.ensemble.forest import RandomForestClassifier
import test

#Generic function for making a classification model and accessing performance:
def classification_model(model, data,data_test, predictors, outcome):
    #Fit the model:
    model.fit(data[predictors], data[outcome])
    
    #Make predictions on training set:
    predictions = model.predict(data[predictors])

    #Make predictions on test set:
    predictions_test = model.predict(data_test[predictors])
    
    preds_df = pd.DataFrame(list(zip(data_test["Loan_ID"], predictions_test)), columns= ['Loan_ID', 'Loan_Status'])
    
    preds_df.to_csv("Final_subs.csv", index=False, doublequote=False)
    #Perform k-fold cross-validation with 5 folds
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    
    for train, test in kf:
        #Filter training data
        train_predictors = (data[predictors].iloc[train, :])
        
        #The target we are using to train the algorithm
        train_target = data[outcome].iloc[train]
        
        #Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)
        
        
        #Record the error for each cross-validation run
        error.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))
        
    print("Cross Validation Score : %s" % "{0:,.3%}". format(np.mean(error)))
        
        #Fit the model again so that it can be referred outside the function
    model.fit(data[predictors], data[outcome])
    
    


#Read the loan train data
loan_train = pd.read_csv("C:\\Users\\ksaldanh\\DataScienceProjects\\Loan_Prediction\\Train.txt", sep=",")

#Read the loan test data
loan_test = pd.read_csv("C:\\Users\\ksaldanh\\DataScienceProjects\\Loan_Prediction\\Test.txt", sep=",")
print(loan_test.head(4))
loan_data =pd.concat([loan_train, loan_test])
loan_data.index = range(0, len(loan_data))

###############Looks like the Dependent column is string in nature, we need to convert it to numeric##########
loan_data.loc[:, 'Dependents'].replace("3+","3", inplace=True)
loan_data["Dependents"] = pd.to_numeric(loan_data["Dependents"])  
#Let's have a look at the summary of numerical variables'
print(loan_data.describe())
   
##############################################EDA################################################
#Let's check what is the difference between those that got their loan approved and those that did not
loan_approval = loan_data.groupby('Loan_Status')
print(loan_approval.mean())

 #Check the number of people that got their loan approved and the number that didn't
print(loan_data.Loan_Status.value_counts())

#Let's visualize our data
features=['Loan_Amount_Term', 'Dependents','Gender', 'Married', 'Credit_History','Property_Area', 'Education']
fig = plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j, data = loan_data)
    plt.xticks(rotation = 90)
    plt.title("Loan Approval")
plt.show()

 #We can observe that credit history in approval or rejection of a loan application, so let's try and understand more closely
temp1 = loan_data['Credit_History'].value_counts(ascending=True)
temp2 = loan_data.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

print ('Frequency Table for Credit History:')
print (temp1)

print ('\nProbility of getting loan for each Credit History class:')
print (temp2)

#Now let us visualize who actually got a loan based on their credit history
temp3 = pd.crosstab(loan_data['Credit_History'], loan_data['Loan_Status'])
temp3.plot(kind = 'bar', stacked = True, color = ['red', 'blue'], grid = False)
plt.show()

#Let us also check how education plays a role
temp4 = pd.crosstab(loan_data['Education'], loan_data['Loan_Status'])
temp4.plot(kind = 'bar', stacked = True, color = ['red', 'blue'], grid = False)
plt.show()

#Let us also check how marital status plays a role
temp4 = pd.crosstab(loan_data['Married'], loan_data['Loan_Status'])
temp4.plot(kind = 'bar', stacked = True, color = ['red', 'blue'], grid = False)
plt.show()

#Check how Loan_Amount_Term plays a role
temp4 = pd.crosstab(loan_data['Loan_Amount_Term'], loan_data['Loan_Status'])
temp4.plot(kind = 'bar', stacked = True, color = ['red', 'blue'], grid = False)
plt.show()


###################################Data Munging#########################################
##############Check for missing values###################################
print(loan_data.apply(lambda x: sum(x.isnull()),axis = 0))

#######################Let's start with LoanAmount and impute missing values with mean###########
loan_data['LoanAmount'].fillna(loan_data['LoanAmount'].mean(), inplace = True)

####################Moving to 'Self_Employed', let's check the trends there######################
print(loan_data['Self_Employed'].value_counts())

################################Since the majority appears to be 'No', it is a safe bet to impute the missing values as 'No'
loan_data['Self_Employed'].fillna('No',inplace=True)

####################Next up 'Loan_Amount_Term', let's check the trends there######################
print(loan_data['Loan_Amount_Term'].value_counts())

################################Since the majority appears to be '360', it is a safe bet to impute the missing values as '360'
loan_data['Loan_Amount_Term'].fillna(360,inplace=True)

####################Next up 'Credit_History', let's check the trends there######################
print(loan_data['Credit_History'].value_counts())

#Since we have observed, loan approval is very dependent on credit history, it's safe to assume those whose loan was rejected had '0' and vice versa
credit_history_boolean = pd.isnull(loan_data['Credit_History'])
loan_data['Credit_History'].fillna(1,inplace=True)

####################Let's check the Married column trends there######################
print(loan_data['Married'].value_counts())

loan_data['Married'].fillna("Yes",inplace=True)


###################Let's check the Dependents column trends there######################
print(loan_data['Dependents'].value_counts())
   
           
#############################Since, from the first plot it can be observed a 0 dependent has an overwhelming probability of a loan approval####################
dependent_boolean = (loan_data['Dependents'].isnull())
#===============================================================================
for i in range(0, len(loan_data)):
     if(loan_data.iloc[i]['Married']=="Yes" and dependent_boolean.values[i]==True):
         loan_data.loc[i:i, 'Dependents'].replace(loan_data.iloc[i]['Dependents'],1.0, inplace=True)
     else:   
         loan_data.loc[i:i, 'Dependents'].replace(loan_data.iloc[i]['Dependents'],0.0, inplace=True)
                
 ####################Let's check the Gender column trends there######################
print(loan_data['Gender'].value_counts())   
#Let us also check how dependents status plays a role in loan approval
temp4 = pd.crosstab(loan_data['Dependents'],loan_data['Loan_Status'])
temp4.plot(kind = 'bar', stacked = True, color = ['red', 'blue', 'green', 'black'], grid = False)
plt.show()              
    
    
###########################3From the plot, it can be observed that people with 2 or more dependents tend to be male################
gender_boolean = pd.isnull(loan_data['Gender'])

for i in range(0, len(loan_data)):
    if(loan_data.iloc[i]['Dependents']>=2 and gender_boolean.values[i]==True):
        loan_data.loc[i:i, 'Gender'].replace(loan_data.iloc[i]['Gender'],"Male", inplace=True)
    else:
        loan_data.loc[i:i, 'Gender'].replace(loan_data.iloc[i]['Gender'],"Female", inplace=True)
    
print("DOme")      
####################################################Let us do some Feature Engineering########################
##################Add the ApplicantIncome and coApplicantIncome to create a new predictor 'TotalIncome'##########################
loan_data["TotalIncome"] = loan_data["ApplicantIncome"] + loan_data["CoapplicantIncome"]
##############################Let us log transform LoanAmount and totalIncome to nullify high values###################
loan_data['LoanAmount_log'] = np.log(loan_data['LoanAmount'])
loan_data['TotalIncome_log'] = np.log(loan_data['TotalIncome'])  
######################Now let's remove ApplicantIncome, coapplicantIncome, LoanAmount and TotalIncome###################

del loan_data["ApplicantIncome"]
del loan_data["CoapplicantIncome"]
del loan_data["LoanAmount"]
del loan_data["TotalIncome"]

print(loan_data.dtypes)  
loan_data['Loan_Status'].fillna('Yes',inplace=True)


###########################Since sklearn requires all variables to be numeric##############
var_mod = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    loan_data[i] = le.fit_transform(loan_data[i])

###########Split into test and train############################
loan_train_new = loan_data.iloc[0:loan_train.shape[0]-1, ]
loan_test_new = loan_data.iloc[loan_train.shape[0]:len(loan_data), ] 

loan_test_new.loc[:, 'Loan_Status'].replace(2,np.nan, inplace=True)


##########################Let us apply Random Forest######################
outcome_var = "Loan_Status"
model = RandomForestClassifier(n_estimators=100)

predictor_var = ['Gender','Dependents','Credit_History','Education','Married','Self_Employed','LoanAmount_log','TotalIncome_log','Loan_Amount_Term']
classification_model(model, loan_train_new,loan_test_new,predictor_var,outcome_var)
    
######################You will observe accuracy is at 100% classic case of overfitting##########
#############Get Feature Importance##############################
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print (featimp)    
    
#####################Let us take the top 5 features###################################
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Loan_Amount_Term','Credit_History','Education']
classification_model(model, loan_train_new,loan_test_new,predictor_var,outcome_var)
