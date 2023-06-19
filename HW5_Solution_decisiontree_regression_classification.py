class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, num):
        self.result += num
        return self.result

    def subtract(self, num):
        self.result -= num
        return self.result

    def multiply(self, num):
        self.result *= num
        return self.result

# Step 1: Identify pronouns to be changed
# - Change all references from plural (e.g., "we") to be individual (e.g., "I")

# Step 2: Replace plural pronouns with individual pronouns
# - Replace all instances of plural pronouns with individual pronouns in the codebase

# Change "we" to "I"
# Change "our" to "my"
# Change "us" to "me"
# Change "ourselves" to "myself"
# Change "We" to "I"
# Change "Our" to "My"
# Change "Us" to "Me"
# Change "Ourselves" to "Myself"

# KEEP_EXISTING_CODE

# coding: utf-8

# # hw5 MSBA-326 Decision Trees using scikit-learn
# #####

# ## 2)
# ### <font color=red>What are the precision and recall definitions? </font>
# 
# >> ##### <font color=green>Precision:</font> 
# The percentage of positive cases predicted correctly out of all of the positive case you predicted as positive. In other words, the accuracy of the positive predictions.
# >> ## $\frac{True Positive}{True Positive + False Positive}$
# >> ##### <font color=green>Recall:</font> 
# The percentage of positive cases predicted correctly out of ALL of the true positive cases.
# >> ## $\frac{True Positive}{True Positive + False Negative}$
# 
# ### <font color=red>Explain why there is a tradeoff between the two? </font>
# 
# > ##### <font color=green>Ans:</font> 
# - If we target higher precision, the denominator will decrease in the precision equation (because __False positives__ decrease) making the output result of the equation increase. In turn, __False Negatives__ will increase. 
# - Because __False Negatives__ increased since we are targeting better precision, the denominator in the recall equation increases making the output result of the recall equation to be decreased.

# ## 3)
# ### <font color=red>What is the definition of __F1 score__ and how do you interpret a high F1 score? </font>
# > ##### <font color=green>Ans:</font> 
# - F1 score is the weighted average of precision and recall
# \begin{equation*}
# {F1} = 
# {2} \times \frac{(precision)(recall)}{precision + recall}\
# \end{equation*}
# 
# > - Since the F1 score is a single metric to measure the average of both precision & recall, I would interpret a high F1 score as performing well in both precision & recall.
# > - The F1 score is a good way to compare estimators.
# 
# 

# ## 4)

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#get_ipython().magic('matplotlib inline')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[2]:


df = pd.read_csv('C:/Users/jahan/Documents/MLData/Baseball_salary.csv')


# #### Cleaning the Data

# In[3]:


df.rename(columns={'Unnamed: 0': 'Names'},inplace=True)
#df.rename(columns={'CAtBat':'CareerAtBat','CHits':'CareerHits','CHmRun':'CareerHR','CRuns':'CareerRuns','CRBI':'CareerRBI','CWalks':'CareerWalks'},inplace=True)


# In[4]:


#testCol.apply(lambda x: x.strip('-'))
df['Names'] = df['Names'].apply(lambda x: x.strip('-'))


# In[5]:


df = df.dropna()


# In[6]:


sns.distplot(df.Salary)


# In[7]:


df['Salary'] = df['Salary'].apply(lambda x: np.log(x))


# In[8]:


plt.figure()
correlation = df.corr(method='pearson')
sns.heatmap(correlation,square=True)


# In[9]:


sns.set(style='ticks')
sns.pairplot(df)


# ## 5)
# #### <font color=red>Descriptive Statistics</font>
# > ##### - Descriptive statistics can be good for seeing if the data set would benefit from standardization.

# In[10]:


df.describe()


# ## 6)
# ### Histogram for each column
# 
# > #### <font color=red>Interpretation of each histogram</font>:
# 
# >> ##### <font color=green>Skewed right features</font> (the majority of values are in the lower range of the distribution):
# >>> - Assists
# >>> - CAtBat
# >>> - CHits
# >>> - CHmRun
# >>> - CRBI
# >>> - CRuns
# >>> - CWalks
# >>> - HmRun
# >>> - PutOuts
# >>> - Years
# >>> - Errors
# 
# >> ##### <font color=green> Gaussian(ish)</font>:
# >>> - AtBat
# >>> - Hits
# >>> - Runs
# >>> - Salary
# >>> - Walks
# >>> - RBI

# In[11]:


plt.figure()
df.hist(alpha=0.5, bins=20,figsize=(10,10));
plt.tight_layout()


# ## 7) Dependent variable + six independent variables:
# 
# __Dependent variable:__
# - log(Salary)
# 
# __Independent variable(s):__
# - Years
# - Hits
# - Homeruns
# - Walks
# - RBI
# - Assists

# ## 8) Decision Tree Regressor

# In[12]:


from sklearn.tree import DecisionTreeRegressor
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus


# In[13]:


df.head()


# In[14]:


feats = ['Years', 'Hits', 'HmRun', 'Walks', 'RBI', 'Assists']
newDF = df[feats]
y = df['Salary']


# In[15]:


regressor = DecisionTreeRegressor()
regressor.fit(newDF, y)


# In[16]:


# use "jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10" 
# for jupyter data rate errors
dot_data = StringIO()
export_graphviz(regressor, out_file=dot_data,
               filled=True, rounded=True,
               special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# ## 9) How many levels does the default decision tree have based on the six features?
# 
# - The default decision tree depth generated for these six features is 15.

# ## 10) Depth 2 decision tree

# In[17]:


regressor2 = DecisionTreeRegressor(max_depth=2)
regressor2.fit(newDF, y)


# In[18]:


dot_data2 = StringIO()
export_graphviz(regressor2, out_file=dot_data2,
               filled=True, rounded=True,
               special_characters=True)
graph2 = pydotplus.graph_from_dot_data(dot_data2.getvalue())
Image(graph2.create_png())


# ### 10) continued:
# > #### <font color=red>Comparing #8 & #10 Decision Trees:
# >> - The decision tree generated in #8 goes deeper level-wise which makes it more susceptible to overfitting/high variance in which the predictions are overly sensitive to "noise".
# >> - The decision tree generated in #10 is a simpler tree which does not suffer from overfitting but can be an over simplified model that introduces bias. Bias or underfitting is the opposite of variance/overfitting where the model is not complex enough for the given data set.
# >> - In the end a machine learning practitioner will want to find the happy medium between high bias and high variance (bias-variance tradeoff).

# ## 11) Depth 3 decision tree
# > #### <font color=red>Comparing #11 decision tree</font>
# >> - Revisiting the explanation in #10, the decision tree generated below is in between (as far as depth) the trees generated in #8 and #10. 
# >> - Depending on the data, this tree maybe the middle ground between high bias and high variance discussed in #10.

# In[19]:


regressor3 = DecisionTreeRegressor(max_depth=3)
regressor3.fit(newDF, y)


# In[20]:


dot_data3 = StringIO()
export_graphviz(regressor3, out_file=dot_data3,
               filled=True, rounded=True,
               special_characters=True)
graph3 = pydotplus.graph_from_dot_data(dot_data3.getvalue())
Image(graph3.create_png())


# ## 12) 
# > <font color=red>If you were to optimize this system using the tree depth (max_depth) as the hyperparameter, what would be your suggestion?</font>
# 
# >> - I would use an automated way to find the optimal depth (grid_search or randomized search).

# ## 13) Bank Note Data set

# In[21]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[22]:


bankDF = pd.read_csv("C:/Users/jahan/Documents/MLData/bill_authentication.csv")


# In[23]:


bankDF.head()


# In[24]:


X = bankDF.drop('Class', axis=1)
y = bankDF['Class']


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.20)


# In[26]:


clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)


# In[27]:


dot_data_clf = StringIO()
export_graphviz(clf, out_file=dot_data_clf,
               filled=True, rounded=True,
               special_characters=True)
graph_clf = pydotplus.graph_from_dot_data(dot_data_clf.getvalue())
Image(graph_clf.create_png())


# ## 14) Recall/Precision curve & Precision/Recall vs. Threshold (Using DecisionTree predict_proba

# In[28]:


from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import roc_curve,roc_auc_score

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(1,figsize=(10,10))
    plt.subplot(211)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Thresholds")
    plt.legend(loc="lower right")
    plt.ylim([0,1])
    
    plt.subplot(212)
    plt.plot(recalls,precisions, alpha=0.2,color='red',lw=5)
    plt.xlim([0.02,0.99])
    plt.ylim([0,1.05])
    plt.grid(True)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout()
    
def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,'b-',label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.xlabel('True Positive Rate')


# In[29]:


clf = DecisionTreeClassifier()
y_scores = cross_val_predict(clf, x_train, y_train, cv=3, method="predict_proba")
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores[:,1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)


# ## 15) Decision Tree - Precision/Recall vs. Threshold: 
# > #### Optimal threshold to achieve a high precision:
# >> - To solely achieve a high precision (ignoring recall) I can pick a threshold in the range of 0.8 to 0.99.
# >> - This would not work for every case because not every model has the sample patterns.
# >> - This is not a good approach to optimizing a classifier because I will most likely be neglicting recall when evaluating the classifier.

# ## 16) ROC curve (for DecisionTree Classifier - pred_proba):

# In[30]:


fpr, tpr, thresholds = roc_curve(y_train,y_scores[:,1])
plot_roc_curve(fpr,tpr)
print('Area Under The Curve: ',roc_auc_score(y_train,y_scores[:,1]))


# ## 17) Using DecisionTrees :
# #### Tree Depth: 2
# > - A suitable threshold to achieve a high precision seems to be at 0.6 to 0.9.
# > - Picking a threshold where precision and recall intersect could be a good strategy for cases where the optimal point of the precision/recall tradeoff can be identified.

# In[31]:


clf2 = DecisionTreeClassifier(max_depth=2)

y_scores2 = cross_val_predict(clf2, x_train, y_train, cv=3, method="predict_proba")
precision2, recall2, thresholds2 = precision_recall_curve(y_train,
                                                      y_scores2[:,1])
plot_precision_recall_vs_threshold(precision2,recall2,thresholds2)


# In[32]:


fpr, tpr, thresholds = roc_curve(y_train,y_scores2[:,1])
plot_roc_curve(fpr,tpr)
print('Area Under The Curve: ',roc_auc_score(y_train,y_scores2[:,1]))


# #### Tree Depth: 3
# > - A suitable threshold to achieve a high precision seems to be between 0.65 & 0.98.
# > - Picking a threshold where precision and recall intersect could be a good strategy for cases where the optimal point of the precision/recall tradeoff can be identified.

# In[33]:


clf3 = DecisionTreeClassifier(max_depth=3)

y_scores3 = cross_val_predict(clf3, x_train, y_train, cv=3, method="predict_proba")
precision3, recall3, thresholds3 = precision_recall_curve(y_train,
                                                      y_scores3[:,1])
plot_precision_recall_vs_threshold(precision3,recall3,thresholds3)


# In[34]:


fpr, tpr, thresholds = roc_curve(y_train,y_scores3[:,1])
plot_roc_curve(fpr,tpr)
print('Area Under The Curve: ',roc_auc_score(y_train,y_scores3[:,1]))
