# Machine Learning Primer


### Introduction
Machine learning is all the rage today.  You can't get far without being reminded of something pertaining to machine learning. Some examples of machine learning that we use every day are Alexa by Amazon, Netflix recommender system for recommending our movies and TV choices, Tesla autopilot self-driving cars and even Splunk’s Chatbot for its sales desk; just to name a few. 

First and foremost, machine learning is a subset of artificial intelligence.  Additionally, deep learning is a subset of machine learning. See image below:

![Artificial Intelligence](https://docs.microsoft.com/en-us/azure/machine-learning/media/concept-deep-learning-vs-machine-learning/ai-vs-machine-learning-vs-deep-learning.png)


### Terminology and Definitions

Let's start with a definition of these three terms.
Wikipedia defines artificial intelligence as follows: “In computer science, artificial intelligence (AI), sometimes called machine intelligence, is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.”<sup>1</sup> Although this term came into existence in the 1960s it is still in wide used today. 

Machine Learning is about 20 years old and is where the bulk of the research and development activities have taking place. Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to carry out tasks without explicit instructions, such as by using pattern recognition and inference.<sup>2</sup>   An explosion of machine learning algorithms was invented and implemented in the early 2000’s making this technology mainstream.  Specifically, most of the past success with AI has taken place in this space. 

Deep Learning is best defined by Jon Krohn in his video course "Deep Learning with TensorFlow".  He states, *“Deep learning involves stacking these straightforward little algorithms called artificial neurons together to solve problems.”*  A neuron as its name implies mimics the human brain physiology. This lines up with Wikipedia definition, "Deep learning is a class of machine learning algorithms that uses multiple layers to progressively extract higher level features from the raw input. For example, in image processing, lower layers may identify edges, while higher layers may identify the concepts relevant to a human such as digits or letters or faces."<sup>3</sup> The topic of deep learning is beyond the scope of this document therefore we will transition to the document's primary focus;  Machine Learning. 

On a higher level, Yufeng G states that Machine Learning has seven steps<sup>4</sup>.  They are:
1. Gathering data
2. Preparing that data
3. Choosing a model
4. Training
5. Evaluation
6. Hyperparameter tuning
7. Prediction



### Machine Learning Process

Spunk has made incredible advancements in the area of machine learning.  Splunk machine learning toolkit it is the best in the business.  According to its website, Splunk implements the following Python packages: *scikit-learn, statsmodels, scipy and custom algorithms.*

The organization ***analyticsvidhya*** compiled the following list of commonly used Machine Learning Algorithms.
[AI Machine Learning Article](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)

Here is a table of commonly used machine learning algorithms with the equivalent Splunk command:


| Article's Algorithm  | Splunk Equivalent Command|
| ----------- | ----------- |
| Linear Regression | \| fit LinearRegression temperature from date_month date_hour into temperature_model 
| Logistic Regression |\| fit LogisticRegression SLA_violation from IO_wait_time into sla_model 
| Decision Tree |\| fit DecisionTreeClassifier SLA_violation from * into sla_model
| SVM |\|  fit SVM SLA_violation from * into sla_model 
| Naive Bayes |
| kNN | \| fit SpectralClustering * k=3 \| stats count by cluster
| K-Means |\| fit KMeans * k=3 \| stats count by cluster
| Random Forest |\| fit RandomForestRegressor temperature from date_month date_hour into temperature_model 
| Dimensionality Reduction Algorithms |\| fit PCA "SS_SMART_1_Raw", "SS_SMART_2_Raw", "SS_SMART_3_Raw", "SS_SMART_4_Raw", "SS_SMART_5_Raw" k=2 into example_hard_drives_PCA_2
| Gradient Boosting algorithm |\| fit GradientBoostingRegressor temperature from date_month date_hour into temperature_model

Most of these algorithms are available in Splunk’s MLTK.

### A Splunk Example

We will spend much of our time addressing and discussing a simple machine learning model implemented in Splunk of open source publicly available data.  These examples are available on the Splunk GitHub site.

In order to understand machine learning, we must first understand the fundamental process of machine learning.   The process is train; fit; apply steps that all machine learning model uses.

First you must train your model on some portion of the dataset which is randomly sampled from about 70% of your original data. This subset of data is called your training data.   Splunk MLTK will give you an option to vary your training data to your specification.  Seventy percent is the recommended size of your training data.  The remainder of your data is called your test data set. 

Below are two examples in Splunk SPL of two machine learning models:

* Example 1: Logistic Regression - 
##### Splunk SPL

```
   | inputlookup churn.csv
   | fit NPR "Churn?" from State into NPRModel
   | fit LogisticRegression fit_intercept=true "Churn?" from NPR* "Day Mins" "Eve Mins" "Night Mins" "Night Charge" 
               "Int'l Plan" "Intl Mins" "Intl Calls" "Intl Charge" "CustServ Calls" "VMail Plan" 
          into ChurnModel
```



* Example 2: Density Function - Anomaly Detection
##### Splunk SPL

```
   | inputlookup call_center.csv
   | eval _time=strptime(_time, "%Y-%m-%dT%H:%M:%S")
   | bin _time span=15m
   | eval HourOfDay=strftime(_time, "%H")
   | eval BucketMinuteOfHour=strftime(_time, "%M")
   | eval DayOfWeek=strftime(_time, "%A")
   | stats max(count) as Actual by HourOfDay,BucketMinuteOfHour,DayOfWeek,source,_time
   | fit DensityFunction Actual by "HourOfDay,BucketMinuteOfHour,DayOfWeek" into mymodel
```

According to Splunk, there are over 30 common algorithms and access to more than 300 popular open-source algorithms through the Python for Scientific Computing library.

Splunk Machine learning toolkit uses a combination assistant helper input forms to aid in the building of machine learning models. They are:

Splunk MLTK Assistants Example :
   * Predict Numeric Fields (Linear Regression): e.g. predict median house values.
   * Predict Categorical Fields (Logistic Regression): e.g. predict customer churn.
   * Detect Numeric Outliers (distribution statistics): e.g. detect outliers in IT Ops data.
   * Detect Categorical Outliers (probabilistic measures): e.g. detect outliers in diabetes patient records.
   * Forecast Time Series: e.g. forecast data center growth and capacity planning.
   * Cluster Numeric Events: e.g. Cluster Hard Drives by SMART Metrics


---

######  Footnotes
   
 <sup>1</sup> https://en.wikipedia.org/wiki/Artificial_intelligence
 
 <sup>2</sup> https://en.wikipedia.org/wiki/Machine_learning
 
 <sup>3</sup> https://en.wikipedia.org/wiki/Deep_learning
 
 <sup>4</sup> https://towardsdatascience.com/the-7-steps-of-machine-learning-2877d7e5548e


