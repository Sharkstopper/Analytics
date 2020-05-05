# Machine Learning Primer rev 1


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

First you must train your model on some portion of the dataset which is randomly sampled from about 70% of your original data. This subset of data is called your training data.   Splunk MLTK will give you an option to vary your training data to your specification.  Seventy percent is the recommended size of your training data.  The remainder of your data is called your test data set. The test dataset is used to validate the model after it trains on the training data.

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
Data sample:


![Head churn Data](/source/images/splunkChurnHead.png)


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

### A Splunk Machine Learning Walkthru for A to Z
As a way of reinforcing the Artificial Intelligence subjects discussed above, this section will use Yufeng’s seven steps of Machine Learning<sup>4</sup> through the application of Splunk’s machine learning and deep learning toolkits to determine if a series of encoded baseball base stealing signals can be interpreted as “Steal” or “No Steal”. 

#### Gathering data
The dataset consists of two comma-separated values (CSV) files containing 5K and 50K rows of data respectively. Each row contains two fields. 
<font size="1">  
| Field 1; ‘STRING’: a sequence of 6 to 20 base stealing signals encoded (‘A’ through ‘G’ )    |Field 2: ‘OUTCOME’ : a  state encoded “STEAL” or “NO”    | 
| :------------- | :---------- | 
|  ABCAEFC | STEAL   | 
|  AAAAAA | NO   | 
</font>  
See the snapshot of the 5K file below:
![image of csv file](/source/images/DataSetExample.png)

#### Preparing that data
The data in the csv files are raw and must be transformed into structure where ML algorithms can operate on them.  For data scientists, data transformation is the most tedious part of the job.  The Analytics India Magazine states that “Data Scientists spend 60% of their time on cleaning and organizing data.”<sup>5</sup>

Other data preparation terms are “data wrangling” and “data munging”.  According to Wikipedia, “Data wrangling, sometimes referred to as data munging, is the process of transforming and mapping data from one "raw" data form into another format with the intent of making it more appropriate and valuable for a variety of downstream purposes such as analytics.”<sup>6</sup> 

The job of cleaning data, data munging or data wrangling gets a lot more manageable with Splunk.  Splunk is an excellent self-contain application for preparing data for machine learning. 

Splunk Dataset loading: To upload the source csv file into Splunk, navigate through following tabs: ‘<Settings>>><Lookups>>> <Lookup Table files>’. Click ‘+Add new’ button.  Next, follow the prompts by selecting 1. destination app, 2. choose local source .csv file and 3. destination filename with a .csv extension.  The destination file is now ready for wrangling using SPL statements to prepare data for a machine learning model.

Let us walk through the data cleaning SPL for this task.
<font size="2">    
```
| inputlookup sign_stealing_raw_fifty.csv
| sample 5000 | eval high = 10 | eval low = 1
| eval STRING = rtrim(STRING)
| eval size = len(STRING)
| eval S001=substr(STRING, 1, 1)  | eval S002=substr(STRING, 2, 1) | eval S003=substr(STRING, 3, 1) 
| eval S004=substr(STRING, 4, 1)  | eval S005=substr(STRING, 5, 1) | eval S006=substr(STRING, 6, 1) 
| eval S007=substr(STRING, 7, 1)  | eval S008=substr(STRING, 8, 1) | eval S009=substr(STRING, 9, 1)
| eval S010=substr(STRING, 10, 1) | eval S011=substr(STRING, 11, 1)| eval S012=substr(STRING, 12, 1)
| eval S013=substr(STRING, 13, 1) | eval S014=substr(STRING, 14, 1)| eval S015=substr(STRING, 15, 1)
| eval S016=substr(STRING, 16, 1) | eval S017=substr(STRING, 17, 1)| eval S018=substr(STRING, 18, 1) 
| eval S019=substr(STRING, 19, 1) | eval S020=substr(STRING, 20, 1)
| fillnull value="Z"
| eval S007 = if(S007="",round(((random()%high)/(high))*(high-low)+low),round(((random() % high)/(high))*(high - low) + low))
| eval S008 = if(S008="",round(((random()%high)/(high))*(high-low)+low),round(((random() % high)/(high))*(high - low) + low))
                              .......
| eval S019 = if(S019="",round(((random()%high)/(high))*(high-low)+low),round(((random() % high)/(high))*(high - low) + low))
| eval S020 = if(S020="",round(((random()%high)/(high))*(high-low)+low),round(((random() % high)/(high))*(high - low) + low)) 
| eval OUTCOME = if(OUTCOME = "NO", 0, 1)
| fields S001 S002 S003	S004 S005 S006 S007	S008 S009 S010	S011 S012 S013 S014	S015 S016 S017 S018	S019 S020  OUTCOME
| replace A WITH 1 IN S001 S002	S003 S004 S005 S006	S007 S008 S009 S010	S011 S012 S013 S014 S015 S016 S017 S018	S019 S020
| replace B WITH 2 IN S001 S002	S003 S004 S005 S006	S007 S008 S009 S010	S011 S012 S013 S014 S015 S016 S017 S018	S019 S020
                                  .......
| replace I WITH 9 IN S001 S002	S003 S004 S005 S006	S007 S008 S009 S010	S011 S012 S013	S014 S015 S016 S017	S01	S019 S020
| replace J WITH 10 IN S001 S002 S003 S004 S005 S006 S007 S008 S009 S010 S011 S012 S013	S014 S015 S016 S017	S018 S019 S020
| replace Z WITH 15 IN S001 S002 S003 S004 S005 S006 S007 S008 S009 S010 S011 S012 S013	S014 S015 S016 S017	S018 S019 S020
| table S001 S002 S003 S004 S005 S006 S007 S008 S009 S010 S011 S012 S013 S014 S015 S016 S017 S018 S019 S020  OUTCOME
```
</font>

The above query produces the following transformed table:

![Splunk Output of transformed data](/source/images/TransformedData.png)
#### Choosing a model

Model selection is the most important step in the machine learning process and extreme diligence must be applied to ensure that the correct model is chosen.  A few words from Splunk Inc regarding Machine Learning and Deep Learning. Splunk gives the following precautions: “The Machine Learning Toolkit is not a default solution, but a way to create custom machine learning. You must have domain knowledge, Splunk Search Processing Language (SPL) knowledge, Splunk platform experience, and data science skills or experience to use the MLTK.” <sup>7</sup> 

Before we proceed with model selection in Splunk, let us look at a tool from the SciPy ecosystem scikit-learn which is the most prevalent machine learning for Python.  Scikit learn provides python packages that puts a wrapper around most major machine learning algorithms.  The Scikit flowchart cheat sheet below is a good place to start for trimming down on which models to select for any machine learning  project. 

[See the scikit learn algorithm cheat-sheet.](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

From the scikit learn cheat sheet we will use the 50K dataset for this example.   **START** `>50 samples`  **yes** *sample size is greater than 50.* Next  `predicting a category` **yes** `the data is labeled` **NO** `<100K samples` **NO**.  Thus as indicated by the cheat sheet, the flow chart suggests a `SGD classifier`. 

"A Stochastic gradient descent (often abbreviated SGD) is an iterative method for optimizing an objective function with suitable smoothness properties (e.g. differentiable or subdifferentiable)."<sup>8</sup>

The image below shows how the SGD model separates out data points into two output classes based on two factors (sepal length and petal length) for the famous iris data set.

[SGD - Stochastic Gradient Descent](https://bogotobogo.com/python/scikit-learn/images/Batch-vs-Stochastic-Gradient-Descent/SGD-Classifier.png)

Now we are ready to look at what model Splunk provides in its MLTK app.

Splunk generalizes the model selection process by presenting its default MLTK screen with general tasks that a Data Scientist want to accomplish such as Predict Fields, Detect Outliers, Forecast Time Series and Cluster Events. Since we want to predict the “Outcome” field based on the transformed fields of S001 through S020,  “Predict Fields” is the best choice.  

Navigate to the Machine Learning Toolkit App which at the time of this writing defaults the showcase dashboard as seen below.

![Machine Learning Capture](/source/images/DJMLTKScreenCapture1.png)
#### Training
    
* Show an example of model training -- here 
    
#### Evaluation
* Evaluationof a Model -- Here   
    
#### Hyperparameter tuning
    
* Small Example here 
![Hyperparameter Splunk Image](/source/images/HyperparameterCapture.png)


#### Prediction
   Deep Learning Dashboard here 
![Machine Learning Capture1](/source/images/BaseBallNNCapture1.png)  
######  Footnotes
   
 <sup>1</sup> https://en.wikipedia.org/wiki/Artificial_intelligence
 
 <sup>2</sup> https://en.wikipedia.org/wiki/Machine_learning
 
 <sup>3</sup> https://en.wikipedia.org/wiki/Deep_learning
 
 <sup>4</sup> https://towardsdatascience.com/the-7-steps-of-machine-learning-2877d7e5548e
 
 <sup>5</sup> https://analyticsindiamag.com/6-tasks-data-scientists-spend-the-most-time-doing/
 
 <sup>6</sup> https://en.wikipedia.org/wiki/Data_wrangling
 
 <sup>7</sup>  https://docs.splunk.com/Documentation/MLApp/5.1.0/User/AboutMLTK
 
 <sup>8</sup>  https://en.wikipedia.org/wiki/Stochastic_gradient_descent
