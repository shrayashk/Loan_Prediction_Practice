# Loan_Prediction_Practice
A classification problem (for the purpose of practicing) where we have to predict whether a loan would be approved or not.
This problem has been attempted using Logistic Regression and associated training and test data has been provided in csv format.

Since this was a fairly quick solution with a rough approach to solving the problem, in hindsight I would have preferred to do Bivariate analysis too, factoring in assumptions and domain information like :

A) Applicants with high income should have more chances of loan approval.
B) Applicants who have repaid their previous debts should have higher chances of loan approval.
C) Loan approval should also depend on the loan amount. If the loan amount is less, chances of loan approval should be high.
D) Lesser the amount to be paid monthly to repay the loan, higher the chances of loan approval.

There are still quite a many things that can be tried to improve the models’ predictions. We can create and add more variables, try different models with different subset of features and/or rows, etc. Some of the ideas are listed below:

A) We can train an XGBoost model and further use grid search to optimize its hyperparameters and improve the accuracy.
B) We can combine the applicants with 1,2,3 or more dependents and make a new feature.
C) We can also make independent vs independent variable visualizations to discover some more patterns.
D) We can arrive at EMI using a better formula which may include interest rates as well.
E) We can even try ensemble modeling (combination of different models).
