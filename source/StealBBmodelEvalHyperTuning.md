```python
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.utils.fixes import loguniform

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
print(__doc__)

import numpy as np

# Name of classifier and an instance of the classifier
classifiers = {
    "Dummy"    : DummyClassifier(strategy='uniform', random_state=2),
    "Log Regression"    : LogisticRegression(),
    "SGD Classifier"    : SGDClassifier(penalty='l1'),
    "Log Regress CV"    : LogisticRegressionCV(),
    "KNN(2)"   : KNeighborsClassifier(2),
    "RBF SVM"  : SVC(gamma=2, C=1),
    "Decision Tree" : DecisionTreeClassifier(max_depth=7),
    "Random Forest" : RandomForestClassifier(max_depth=7, n_estimators=10,max_features=4),
    "Neural Net"  : MLPClassifier(alpha=1),
    "AdaBoost"   : AdaBoostClassifier(),
    "Naive Bayes" : GaussianNB(),
    "QDA"    : QuadraticDiscriminantAnalysis(),
    "LinearSVC" : LinearSVC(),
    "LinearSVM" :SVC(kernel="linear"),
    "Gaussian Proc" : GaussianProcessClassifier(1.0 * RBF(1.0)),
}
```

    Automatically created module for IPython interactive environment
    


```python
input_file = r"C:\Users\gdjsh\Dropbox\code\python\data\baseBallStealData000111.csv"
```


```python
df = pd.read_csv(input_file)
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S001</th>
      <th>S002</th>
      <th>S003</th>
      <th>S004</th>
      <th>S005</th>
      <th>S006</th>
      <th>S007</th>
      <th>S008</th>
      <th>S009</th>
      <th>S010</th>
      <th>...</th>
      <th>S012</th>
      <th>S013</th>
      <th>S014</th>
      <th>S015</th>
      <th>S016</th>
      <th>S017</th>
      <th>S018</th>
      <th>S019</th>
      <th>S020</th>
      <th>OUTCOME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1183</th>
      <td>5</td>
      <td>5</td>
      <td>10</td>
      <td>8</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>7</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>7</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1184</th>
      <td>8</td>
      <td>6</td>
      <td>5</td>
      <td>4</td>
      <td>10</td>
      <td>5</td>
      <td>7</td>
      <td>7</td>
      <td>4</td>
      <td>4</td>
      <td>...</td>
      <td>4</td>
      <td>9</td>
      <td>6</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
      <td>9</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1185</th>
      <td>9</td>
      <td>8</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>6</td>
      <td>9</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>...</td>
      <td>4</td>
      <td>1</td>
      <td>9</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1186</th>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>10</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>6</td>
      <td>...</td>
      <td>9</td>
      <td>7</td>
      <td>5</td>
      <td>3</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>8</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1187</th>
      <td>10</td>
      <td>4</td>
      <td>10</td>
      <td>1</td>
      <td>8</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>6</td>
      <td>4</td>
      <td>...</td>
      <td>9</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
dfcolumns = ["S001","S002","S003","S004","S005","S006","S007","S008","S009","S010",
"S011","S012","S013","S014","S015","S016","S017","S018","S019","S020"]
X,y = df[dfcolumns], df["OUTCOME"]
X.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S001</th>
      <th>S002</th>
      <th>S003</th>
      <th>S004</th>
      <th>S005</th>
      <th>S006</th>
      <th>S007</th>
      <th>S008</th>
      <th>S009</th>
      <th>S010</th>
      <th>S011</th>
      <th>S012</th>
      <th>S013</th>
      <th>S014</th>
      <th>S015</th>
      <th>S016</th>
      <th>S017</th>
      <th>S018</th>
      <th>S019</th>
      <th>S020</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1183</th>
      <td>5</td>
      <td>5</td>
      <td>10</td>
      <td>8</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>9</td>
      <td>7</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>7</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1184</th>
      <td>8</td>
      <td>6</td>
      <td>5</td>
      <td>4</td>
      <td>10</td>
      <td>5</td>
      <td>7</td>
      <td>7</td>
      <td>4</td>
      <td>4</td>
      <td>9</td>
      <td>4</td>
      <td>9</td>
      <td>6</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
      <td>9</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1185</th>
      <td>9</td>
      <td>8</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>6</td>
      <td>9</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>9</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1186</th>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>10</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>6</td>
      <td>5</td>
      <td>9</td>
      <td>7</td>
      <td>5</td>
      <td>3</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>8</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1187</th>
      <td>10</td>
      <td>4</td>
      <td>10</td>
      <td>1</td>
      <td>8</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>6</td>
      <td>4</td>
      <td>1</td>
      <td>9</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
y.tail()
```




    1183    1
    1184    1
    1185    1
    1186    1
    1187    1
    Name: OUTCOME, dtype: int64




```python
X_train,X_test,y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=7)
```


```python
from time import time
nfast = 15
head = list(classifiers.items())[:nfast]

for name, classifier in head:
    start = time()
    classifier.fit(X_train, y_train)
    train_time = time() - start
    score = classifier.score(X_test, y_test)
    score_time = time()- start
    print("{:<15}| score = {:.3f} | time = {:,.3f}s/{:.3f}s".format(name, score, train_time, score))
```

    Dummy          | score = 0.538 | time = 0.001s/0.538s
    Log Regression | score = 0.543 | time = 0.019s/0.543s
    SGD Classifier | score = 0.504 | time = 0.009s/0.504s
    Log Regress CV | score = 0.543 | time = 0.147s/0.543s
    KNN(2)         | score = 0.513 | time = 0.004s/0.513s
    RBF SVM        | score = 0.496 | time = 0.038s/0.496s
    Decision Tree  | score = 0.594 | time = 0.004s/0.594s
    Random Forest  | score = 0.555 | time = 0.016s/0.555s
    Neural Net     | score = 0.507 | time = 0.963s/0.507s
    AdaBoost       | score = 0.580 | time = 0.094s/0.580s
    Naive Bayes    | score = 0.515 | time = 0.002s/0.515s
    QDA            | score = 0.541 | time = 0.009s/0.541s
    LinearSVC      | score = 0.518 | time = 0.090s/0.518s
    LinearSVM      | score = 0.557 | time = 0.164s/0.557s
    Gaussian Proc  | score = 0.487 | time = 0.451s/0.487s
    


```python
# get some data
#X, y = load_digits(return_X_y=True)

# build a classifier
clf = SGDClassifier(loss='hinge', penalty='elasticnet',
                    fit_intercept=True)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {'average': [True, False],
              'l1_ratio': stats.uniform(0, 1),
              'alpha': loguniform(1e-4, 1e0)}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {'average': [True, False],
              'l1_ratio': np.linspace(0, 1, num=10),
              'alpha': np.power(10, np.arange(-4, 1, dtype=float))}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
```

    RandomizedSearchCV took 0.84 seconds for 20 candidates parameter settings.
    Model with rank: 1
    Mean validation score: 0.539 (std: 0.028)
    Parameters: {'alpha': 0.6041647468808398, 'average': False, 'l1_ratio': 0.18187005087683672}
    
    Model with rank: 2
    Mean validation score: 0.525 (std: 0.024)
    Parameters: {'alpha': 0.002639301939623978, 'average': False, 'l1_ratio': 0.4601710045535715}
    
    Model with rank: 3
    Mean validation score: 0.520 (std: 0.037)
    Parameters: {'alpha': 0.4795153103415524, 'average': False, 'l1_ratio': 0.1080959849538019}
    
    GridSearchCV took 4.17 seconds for 100 candidate parameter settings.
    Model with rank: 1
    Mean validation score: 0.537 (std: 0.022)
    Parameters: {'alpha': 0.1, 'average': True, 'l1_ratio': 0.0}
    
    Model with rank: 2
    Mean validation score: 0.537 (std: 0.022)
    Parameters: {'alpha': 1.0, 'average': False, 'l1_ratio': 0.0}
    
    Model with rank: 3
    Mean validation score: 0.535 (std: 0.032)
    Parameters: {'alpha': 0.1, 'average': False, 'l1_ratio': 0.2222222222222222}
    
    


```python

```
