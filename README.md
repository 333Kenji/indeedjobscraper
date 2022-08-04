![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/dashboardPrev.jpg "Final")

# Multi-Class Linear Regression on Indeed Salaries

## Introduction
This project is an overhaul of a major project I'd completed last year before Indeed.com started including salary estimates for the most of job postings that didn't have that information.

In this most recent version, I conduct linear regression to predict the salary ranges for job postings that I've webscraped from Indeed.com

I'm revisiting the project because it presents a set of challenges: webscrapping, extensive data wrangling, natural language processing, and multi-label classification employing multiple models.

Below you will find an in-depth guide to this project, however, I do recommend the jupyter notebooks located in the docs folder. These expanded versions ~~have the same code contained in the application itself but~~ provide all visualizations along with dynamic statistical explanations drawn directly from the most recent scraping.

## Project Summary

The overall objective of this project is to create an interactive 'Data Science Job Market' dashboard in Tableau for the visual and statistical analysis conducted by job seekers and recruiters alike.
#insert image of the dashboard
<br><br>
The webscrapped dataset comprises a robust feature set; containing numeric, categorical, and string values. Employing the appropriate approaches to each feature type, I wrangle, engineer, analyze, and transform each before moving on to bi and multivariate feature analysis.
Considering the user, there are several outputs I'd like this product to provide; trend analysis, most common states or key skills/terms, and perhaps most importantly - salary.
<br><br>
I've observed, over the past year working with this data source that only 10-20% of job posts have a salary attached. However, in the time since I started the project, Indeed.com has begun to include salary estimates derived using similar methods found in this project.
<br><br>
I'm combining this addition of 40-60% of salary values with those provided by employers - they're all estimates.
Since this is an exploration of multi-class classification I take the quartile ranges of all the observed (estimated and given) salaries and assign each data point to a target value corresponding to the range (positive class) the salary falls within.
<br><br>
Optimal preprocessing and model hyperparameters needed for each target (quartile/positive class) /model are developed using a combination of GridSearchCV and pipelines from sklearn. These settings are then transferred to a new pipeline where I train, test, and evaluate each model before extracting the probabilistic outputs underlying the otherwise binary classifications for each class. To arrive at the final prediction I compare the probabilities for each model's prediction and select the class with the strongest probability.


## Project Background

I originally conceived this project a year ago, when I and my cohort in the Galvanize data science bootcamp were about to enter the job market as freshers. At that time a major concern of ours was how to determine a job's potential salary based on such factors as required skills and profiencies, location, and description.

After graduating from Galvanize's Data Science bootcamp I immediately sought to build a relevant and useful data science application in order to apply what I'd learned to solve a real-world problem. At that time, the greatest challenge faced by my fellow alums and I was, of course, finding a job. Part of that process involves searching through and filtering job postings based on their descriptions, requirements, responsibilities, location, and of course, potential salary. Unfortunately job listing sites like Indeed.com, despite possessing some advanced search functionality, often place their job market analytics behind a premium membership or paywall. Additionally, most employers do not include salary information in their job descriptions. In fact, through my analysis I found that only 10% of Indeed.com search results for 'data scientist' contain any salary information and without that metric it is difficult to both compare jobs and conduct an analysis of this paticular job market.


# Major Challenges

- Unfortunately, at that time, only 10% or so of data science roles listed on Indeed.com provided a salary. Thus, we had very small samples from which to discern a wide and evolving range of salaries.
Goals: Business Objectives
- The business objective of this project continues to be the successful implementation of multi-label classification to provide salary ranges for data that have not been provided these values either by the emplyer or through Indeed.coms estimates.
Goals: Project Objectives
- Reinforcing particular skills and intuition by tackling data scraping and wrangling, NLP, and multi-label classification.
The Solution
- How this project solves the problem.
- Evaluation Metrics. (if the model can predict with 99% accuracy...)
- Minimum viable product.
    - An accurate classifier.
# Packges & Libraries
- Python
- Numpy, Pandas
- Scikit-learn
- Jupyter, VSCde
# Methods


Using multivariate logistic regression can be used to generate models capable of providing accurate probabilities for each of four salary ranges representing the quartiles observed in the range of observed salaries (The 10% of records that do include salary information). The range with the greatest likelihood is then assigned as that record's classification, (posts with salaries are automatically classed) ensuring that 100% of the records contain salary data.




To accomplish this, I must first provide some sort of salary data for the 90% of records missing that information. Using Natural Language Processing I transform the information found in job postings web scraped from Indeed.com into a dataset.


- Machine Learning
    - I originally chose logistic regression because I wanted a binary answer to wether or not a particular job could be predicted to be above or below the median salary seen in similar observations.
    - Through further investigation of the target's descriptive statitics I eventually divided the observed range of salaries into quartiles, fitting a classifier to each, and instead of going with binary outcomes, compared the probabilistic outcomes, selecting the highest as the most likely and thus selected that probability's associated class as the final classification.
- NLP
    - Parsing each postings HTML and Javascript into features buiolt a table whose values were primarily in string format. While feture like location could be split into city/state features and then encoded for analysis along their descriptive statistics, the summaries and descriptions of each job posting required a full application of natural language processing.
- Data Visualization
- Predictive Modeling
- etc.


# Model
Since I'd already used ensemble methods like random forests to solve classification challenges in several of my Galvanize projects I decided to work with an algorithm that I'd not yet incorporated into a large project. Logistic regression, which leans on  statistical likelihood as opposed to probability serves as an excellent classifier in this case especially since the weighted importance of term that appears in each posting, relative to their appearance across all postings are given by the
transformer component of the TfidfVectorizer algorithm as scaled floating-point values.
The logistic regression model I used incorporated cross-validation which boosts the fidelity of the model's results by providing an average of its performance and is capable of handling mixed data to some extent. I mention this because my final set of features is actually comprised of two tables, the first being the importance of each term as mentioned previously, and the second being a set of binary features that would serve as strong indicators for my classification. For example, if Washinton, NYC, and Texas appear in rows most associated with a particular salary range, and if they appear in a particular posting, the binary feature representing the state would be labeled as 1.0, and 0.0 if not. Depending on the amount of data I had at any given time, the tfid features were many times greater than my set number of 14 binary features; currently they stand at 90+ features, one for each of the most important terms that appear throughout the corpus of job postings.

![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/preprocessedFeatures.jpg "Extracted Features")

At this point, I needed to make a few design choices for the project. I could very well have conducted linear regression which works for predicting continuous values, i.e. specific dollar amounts. Or, I could have used logistic regression or another classifier that could predict for multiple categories since I was already working towards salary brackets as opposed to dollar amounts. For the purpose of this project, a specific dollar amount is neither relevant nor is it useful. Salary negotiations simply don't start with a dollar amount. Any dollar amount could and should be treated as a starting point indicating an expected range to work with. In the case of using a multi-classifier, I wanted to make sure that I could glean as much information about my data and the algorithm's performance as possible for each salary bracket. In particular, the words associated with each range so they could be conveyed to the user via the app if they chose to use the filters I built into the dashboard.

To accomplish this, I actually conduct linear regression several times by building a target variable for each salary bracket and in each iteration calling that particular column as the dependent variable. Also, to expand upon this project in the future I wrapped the entire process of splitting, featurization (binary and tf-idf), a dummy test, and even a decision tree classifier to distill the most important terms even further into a single function that can incorporate new data as I continue to expand this collection of 5000+ rows of data in the coming months. The algorithms used are isolated and discarded after each iteration and will perform better as I add to my collection of scrapped data.

# Metrics

# Project Description
### 0. Webscraping
## 1. Data
## The Data
The data consists of text scraped from every search result for 'data science/scientist' on Indeed.com using the [Requests](https://docs.python-requests.org/en/master/ "Requests Library"), Tor, and [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/ "BeautifulSoup") libraries.

Initially, I'd considered using Indeed.com's API for this project since I've had experience working with them for several of my Galvanize projects (stock market analysis) and an MMO whose developers allow their players to access data in real-time ([EVE Online API](https://esi.evetech.net/ui/ "EVE Online API")). APIs are easy and quick to work with so long as you stay under the rate limit. Unfortunately, the documentation for Indeed's API is rather incomprehensible and due to some recent change on their end and may not even be available to the general public. So instead, I chose to try my hand at web scraping which, as it turns out, can be extremely tedious, nuanced, yet highly rewarding due to the surgical precision nature of extracting data via the HTML tags and other markers present in every web page.


![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/htmlInspect.jpg "Browser Inspection Shows HTML Structure")

The process of reading in a webpage and selecting my data using Requests and BeautifulSoup was a straightforward one. A formatted string containing 'data science/scientist'+location is passed to Indeed.com as a get request which returns a response containing the text representation of the search results. Scanning the HTML for particular fields like 'jobTitle', 'companyName', and 'companyLocation' I was able to build a table of 25000+ rows of data represented by a total of ten initial features.

![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/response.jpg "The Specific HTML Fields We're Gleaning From")

A major hurdle in web scraping is doing so undetected. Make too many queries, at a high frequency, and you may very well get IP banned for the day, or worse. As with rate limits for APIs (how many queries one can make in a given interval), IP banning is used to limit the traffic a website's server must accommodate. For example, in a distributed denial-of-service attack (DDoS), a website is taken down by directing hundreds or thousands of machines to simultaneously try to access a website, overwhelming its servers and causing it to crash.
Although web scraping is generally frowned upon because a single computer can make thousands of requests every minute, a good practice is to play nice and space out the requests a bit. To do this I added a random delay between requests, at a rate of anywhere between 1 and 3 seconds. Additionally, I used the tor library (yes, that tor) to mask my PC’s identity in case even my delayed requests were noticed by Indeed's server monitors. And even so! In order to keep my data up to date I also employ a VPN to shift my IP around whenever it's blocked.
Because my requests were sometimes for thousands of posts at once, and interruptions and 24hr bans did occur with some frequency, I opted to store the data in .csv format (similar to an excel spreadsheet) so I could have a "hard copy" in case my web scraping was interrupted.

And alas, perhaps the greatest hurdle of all`

![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/scraped.jpg "The Specific HTML Fields We're Gleaning From")


However, only 10% of these job postings contain salary information, severely limiting the scope of analysis with the data as-is.

![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/imbalanced.jpg "It's remained at about 10% since start of project")

I'll also take this into account when applying scoring metrics and will go into further detail in the model evaluation section below.
### 1.1 Table Inspection
### 1.2 Data Diagnosis
## 2 Data Wrangling & Feature Engineering



### 2.1 Feature & Task Overview
Since the entirety of my data was in text format, including dates posted and salary information, I had to conduct extensive cleaning and reformating. This step entailed removing unnecessary spaces and characters for most features while isolating and converting salary information given in string format to floating-point values.
Dealing with Indeed's date formatting was another tricky area: search results are dated relative to the day of inquiry, which means that the date posted was given as 1-30+ 'days ago'. Fortunately, I was able to convert values in the 'ExtractDate' to date objects from which I simply subtract the number of days given in the 'Post Date' column. This column would be dropped before conducting the machine learning process but attached afterward in order to provide for temporal visual analysis in the application itself.
While splitting the 'Location' column into 'State' and 'City' features was simple enough, Salary' was a challenging column to convert into a useful feature. After dropping special characters and extracting the given rate of pay (hourly, weekly, monthly, yearly) I converted the string representation of the numeric values given as dollar amounts and extrapolated that information with the rate of pay so that the few salaries that were given were at least annualized. This was crucial because Salary is what I use as my target when training the linear regression algorithm for class (quartile range).

![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/munged.jpg "After Cleaning and Sorting Features")

### 2.1a Detecting Outliers
Once the data had been cleaned and my features preprocessed or converted I was able to conduct some exploratory data analysis of the given salaries; laying out their distribution and identifying and dropping outliers as needed. 

![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/outliers.jpg "With Outliers")
(with outliers)

For for the latter I combined two methods, the Z-Score and quartiles in order to build a list of salaries that first fell outside of 3 standard deviations from the mean, and then finding any additional salaries that were 1.5 times greater or less than the upper and lower bounds of the interquartile range respectively. Any job postings with these dollar amounts were then dropped.

![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/trimmedoutliers.jpg "Trimmed Outliers")

### 2.1b Z-Score
### 2.1c IQR Method
### 2.2 Rating
### 2.3 Job Type
### 2.4 Dates
### 2.5 Company & Job Title
### 2.6 Location
### 2.7 EDA

 ---------- I think this stuff nees to be reordered, in te notebook too. Knock out numerics along with outliers, slip in dates so we can see a pretty timeline, then categoricals [job_type,company,job_title (maybe merge both job features), and location], then target, then break out NLP followed by full EDA


To apply this categorization to the remaining job postings I used sklearn's implementation of [TF-IDF vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html "Term Frequency-Inverse Document Frequency") to extract the top and bottom 30% of words and phrases associated with each quartile, as features in a new table where each posting is represented by the TF-IDF scores of its terms.

![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/Q3words.jpg "Q3 Words")



![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/preprocessedFeatures.jpg "Extracted Features")

For each class this generally produces 90 features (terms) but I also added fourteen static binary features determined by the presence, or absence, of any of the top or bottom 30% of terms.


At this point I've abstracted the text-based data into numerically represented data which is the format required for linear regression. However, in order to predict for all four target labels, while also extracting their associated terms and importance scores, I needed to employ a [One-vs-All](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html "one-vs-all/rest") strategy. In this implementation of that strategy, I iterate over the data using linear regressinon as a binary classifier for each of the quartile classes, taking the highest scoring probability as the likely class to assign each job posting.

### 2.8 Salary
### 2.9 Target
To solved this problem I created four classes based on the quartiles of those postings with salaries, classed as Q1, Q2, Q3, and Q4. Sorting the posts with salaries by the value of that feature, and splitting that series first at the median, then again at the medians adjacent to the median, resulting in four ordered groups in  
![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/quartiles.jpg "Original Data Split By Quartile")

![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/a.jpg "Extracted Features")
## 3. EDA
### 3.1 Bivarirate
### 3.2 Multivariate
## 4. Model Building
### 4.1 Hyoerparameter Tuning with GridSearch Pipeline
### 4.2 Modelling

![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/b.jpg "Likelihoods")


Here, 'colmax' represents the most likely class out of all four.

![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/d.jpg "Likelihoods")
To summarize the table, Q serves as the final verdict - the most likely quartile and 'Probability' is the probability of that particular trial.


![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/final.jpg "Final")
Finally, the tables with given and predicted salary ranges were concatenated so the dashboard user could analyze and filter all of the job postings by location, company, salary bracket, and term relevance. Notice that this is our original, cleaned, data but now we have a salary range for every posting.

## 5. Evaluations
### 5.1 Results Table
![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/final.jpg "Final")
Finally, the tables with given and predicted salary ranges were concatenated so the dashboard user could analyze and filter all of the job postings by location, company, salary bracket, and term relevance. Notice that this is our original, cleaned, data but now we have a salary range for every posting.


Working with multiple models provides for a great deal of parameter tuning possibilities. For example, using just the bare minimum parameters that would be suited for this problem I was able to predict only 42% of the missing salaries but after tuning the logistic regression and tf-idf models I was able to expand that to 65%. And with 10% of the postings containing salary data I have salary ranges for 76% of the data. This is the score I'll be improving over time as I collect more data and continue to improve the models.

In addition to returning the enhanced search results, the machine learning function also produces performance metrics for each salary range prediction as well as their associated word lists.

![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/c.jpg "Likelihoods")
Here, 'colmax' represents the most likely class out of all four. 'Q1 Posts' - 'Q4 Posts' combine both the predicted and given classifications.

![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/d.jpg "Likelihoods")
To summarize the table, Q serves as the final verdict - the most likely quartile and 'Probability' is the probability of that particular trial.



### 5.2 Model Evaluations


![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/confusion_roc.jpg "Confusion & ROC for Test sets")
Overall I'm pleased with the current metrics, despite the dip in correctly labelling Q3 classes. It's also worth noting the strength and similarity of Q1 and Q4. Since this data is inherently imbalanced I'm not too concerned with accuracy as I am about precision and recall, particularly for Q2 and Q3. Also, these classifications will be further refined selecting between the greatest probabilities amongst all 'Q' classes.


Here's a preview of the [Interactive Tableau Dashboard](indeedwebapp-env.eba-qt8deefm.us-east-2.elasticbeanstalk.com/ "Advanced Job Search") I built as a flask application (app folder) which is deployed to an EC2 instance which automatically reads code updates pushed to my git repository, check it out!

![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/dashboardPrev.jpg "Final")


![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/dashboardBott.jpg "Final")






??? data.head() img

## Authors

- Kenji Alford [GIT](https://www.github.com/333kenji)

##### Data Dictionary
- company<br>
&nbsp;
&nbsp;name of company
- description<br>
&nbsp;full text description of job
- estimated_salary<br>
&nbsp;Indeed's salary estimation
- extractDate<br>
&nbsp;String(?), date record was webscrapped
- job_type_items<br>
&nbsp;formatted list of 'full-time', 'part-time', etc
- location<br>
&nbsp;state and city/remote
- postDate<br>
&nbsp;date job posting was created
- rating<br>
&nbsp;Indeed.com ascribed company ratings
- requirements<br>
&nbsp;employer-listed educational, aptitude, and experience requirements
- sal_guide_items<br>
&nbsp;formatted list containing redundant characters and the Indeed salary estimate
- salary<br>
&nbsp;if given by employer
- salary_and_jType<br>
&nbsp;formatted list combining salary from salfromsection and job_type_items
- summary<br>
&nbsp;brief summaries of each job extracted from sach results
- title<br>
&nbsp;job position title
- url<br>
&nbsp;job posting/description address

---







Information About Model
Model Evaluation
    - [Model Card](https://arxiv.org/pdf/1810.03993.pdf)
Predictions
Real World Applications

---
<!-- ROADMAP -->
## Roadmap
### Delivering Analysis
Tieing things back to my interest in providing interactive front-end experiences I built the deployment of this project as a FlaskAPI application. Like Flask, this framework allows for a great deal of control over the look and feel of a webpage, alllowing for complete control over python, javascript, CSS, HTML, and makes use of the Jinja language which I find extremely useful for making modular websites. What sets FastAPI apart as an extension of Flask is that it also allows for APIs to be developed so users can interact with a database through the forward-facing interface of a website.
![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/dashboardPrev.jpg "Final")


![alt text](https://github.com/333Kenji/Machine-Learning-Indeed-Search/blob/main/app/static/images/dashboardBott.jpg "Final")
This will be great to expand up in the future, but for the sake of time and simplicity, I decided to build a dashboard using Tableau which is a platform I'd previously had no experience with unlike Flask. This has allowed me to quickly and a daresay successfully translate my analysis into a useful or at least interesting tool that users can easily manipulate as they explore the data.

- [x] Function to deliver output for Tableau
- [x] Consolidate Main Project into single notebook
- [ ] Handle all technical explanation
- [ ] Images
    - [ ] Ensure Al Images are correctly labelled etc
    - [ ] Adjust/apply custom plot image saver
- [ ] Tidy bi/multivariate EDA