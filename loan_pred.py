import pandas as pd 
import numpy as np                     # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs 
%matplotlib inline 
import warnings                        # To ignore any warnings 
warnings.filterwarnings("ignore")

train=pd.read_csv("train.csv") 
test=pd.read_csv("test.csv")

train_original=train.copy() 
test_original=test.copy()

**********************************************UNIVARIATE ANALYSIS*******************************************************
# We will first look at the target variable, i.e., Loan_Status. As it is a categorical variable, let us look at its frequency table, percentage distribution and bar plot.

# Frequency table of a variable will give us the count of each category in that variable

train['Loan_Status'].value_counts()
train['Loan_Status'].value_counts(normalize=True)
train['Loan_Status'].value_counts().plot.bar()

#Now lets visualize each variable separately. Different types of variables are Categorical, ordinal and numerical.

# Categorical features: These features have categories (Gender, Married, Self_Employed, Credit_History, Loan_Status)
# Ordinal features: Variables in categorical features having some order involved (Dependents, Education, Property_Area)
# Numerical features: These features have numerical values (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term)

plt.figure(1) plt.subplot(221) train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 
plt.subplot(222) train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 
plt.subplot(223) train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 
plt.subplot(224) train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 

# It can be inferred from the above bar plots that:

# 80% applicants in the dataset are male.
# Around 65% of the applicants in the dataset are married.
# Around 15% applicants in the dataset are self employed.
# Around 85% applicants have repaid their debts.
# Now let’s visualize the ordinal variables.

plt.figure(1) plt.subplot(131) train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents') 
plt.subplot(132) train['Education'].value_counts(normalize=True).plot.bar(title= 'Education') 
plt.subplot(133) train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area') 
plt.show()

# Following inferences can be made from the above bar plots:

# Most of the applicants don’t have any dependents.
# Around 80% of the applicants are Graduate.
# Most of the applicants are from Semiurban area.

#Lets look at the distribution of Applicant income first.

plt.figure(1) plt.subplot(121) sns.distplot(train['ApplicantIncome']); 
plt.subplot(122) train['ApplicantIncome'].plot.box(figsize=(16,5)) 
plt.show()

#Let’s look at the Coapplicant income distribution.

plt.figure(1) plt.subplot(121) sns.distplot(train['CoapplicantIncome']); 
plt.subplot(122) train['CoapplicantIncome'].plot.box(figsize=(16,5)) 
plt.show()

#Let’s look at the distribution of LoanAmount variable.

plt.figure(1) plt.subplot(121) df=train.dropna() sns.distplot(train['LoanAmount']); 
plt.subplot(122) train['LoanAmount'].plot.box(figsize=(16,5)) 
plt.show()

**************************FILLING MISSING VALUES AND OUTLIER ADJUSTMENT*****************************************************

train.isnull().sum()
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

train['Loan_Amount_Term'].value_counts()

#It can be seen that in loan amount term variable, the value of 360 is repeating the most. So we will replace the missing values in this variable using the mode of this variable.

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

#We will use median to fill the null values as earlier we saw that loan amount have outliers so the mean will not be the proper approach as it is highly affected by the presence of outliers.

train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

train.isnull().sum()

test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

#Filling all the missing values in the test dataset too using the same approach

train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log'] = np.log(test['LoanAmount'])

******************************************MODEL CONSTRUCTION**********************************************************

train=train.drop('Loan_ID',axis=1) 
test=test.drop('Loan_ID',axis=1)

#Sklearn requires the target variable in a separate dataset. So, we will drop our target variable from the train dataset and save it in another dataset.

X = train.drop('Loan_Status',1) 
y = train.Loan_Status

#Now we will make dummy variables for the categorical variables. Dummy variable turns categorical variables into a series of 0 and 1, making them lot easier to quantify and compare. Let us understand the process of dummies first:

# Consider the “Gender” variable. It has two classes, Male and Female.
# As logistic regression takes only the numerical values as input, we have to change male and female into numerical value.
# Once we apply dummies to this variable, it will convert the “Gender” variable into two variables(Gender_Male and Gender_Female), one for each class, i.e. Male and Female.
# Gender_Male will have a value of 0 if the gender is Female and a value of 1 if the gender is Male.
X=pd.get_dummies(X) 
train=pd.get_dummies(train) 
test=pd.get_dummies(test)
# Now we will train the model on training dataset and make predictions for the test dataset. But can we validate these predictions? One way of doing this is we can divide our train dataset into two parts: train and validation. We can train the model on this train part and using that make predictions for the validation part. In this way we can validate our predictions as we have the true predictions for the validation part (which we do not have for the test dataset).

# We will use the train_test_split function from sklearn to divide our train dataset. So, first let us import train_test_split.

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)

# The dataset has been divided into training and validation part. Let us import LogisticRegression and accuracy_score from sklearn and fit the logistic regression model.

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
model = LogisticRegression() 
model.fit(x_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,          penalty='l2', random_state=1, solver='liblinear', tol=0.0001,          verbose=0, warm_start=False)
# Here the C parameter represents inverse of regularization strength. Regularization is applying a penalty to increasing the magnitude of parameter values in order to reduce overfitting. Smaller values of C specify stronger regularization. To learn about other parameters, refer here: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

# Let’s predict the Loan_Status for validation set and calculate its accuracy.

pred_cv = model.predict(x_cv)
Let us calculate how accurate our predictions are by calculating the accuracy.

accuracy_score(y_cv,pred_cv)
# 0.7945945945945946
# So our predictions are almost 80% accurate, i.e. we have identified 80% of the loan status correctly.

# Lets make predictions for the test dataset.

pred_test = model.predict(test)