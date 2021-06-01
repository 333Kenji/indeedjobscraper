
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pd.options.mode.chained_assignment = None  # default='warn'


""" Jupyter
# Display plots in the notebook
#%matplotlib inline 

# increase default figure and font sizes for easier viewing
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 17
"""


def split_gen(col, data):# Seperating out listing with salary info
    """[Generates train/test splits of data with Given salaries as determined by the target given as col]

    Args:
        col ([string]): [our y target]
        data ([DataFrame]): [munged data]

    Returns:
        [tuple]: [train and test splits]
    """
    salary_data = data[data[col].notnull()]
    # Seperating out listing with salary info.
    sal_X = salary_data[['JobTitle', 'City', 'State', 'Company', 'Requirements', 'Summary', 'Description']]
    sal_X.reset_index(drop=True, inplace=True)

    # Using col to designate the target vector.
    sal_y = pd.DataFrame(salary_data[col])
    sal_y.reset_index(drop=True, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(sal_X, sal_y, test_size=0.2, stratify=sal_y, random_state=74) 

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    return X_train, X_test, y_train, y_test



def rl(col, data):
    """[Performs the full ML process for RL with tfid. To generate predictions for each quartile
    of the given salaries this function is called for each vector representing the quartile the
    salaries are in.]

    Args:
        col ([string]): [our target]
        data ([DataFrame]): [munged data]

    Returns:
        [csv]: [table containing data plus predictions for each quartile of all
        given salaries for that range.]
    """

    X_train, X_test, y_train, y_test = split_gen(col, data)
    def tfid_words(feature):
        """[Builds word/phrase lists and provides numeric values for their frequency.]

        Args:
            feature ([string]): [features containing string values that can be processed using tfid]

        Returns:
            [list]: [full list of words and phrases]
        """
        tvec_feature = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.05, max_features=15)
        tvec_feature.fit(X_train[feature])
        feature_train = pd.DataFrame(tvec_feature.transform(X_train[feature]).todense(),
                           columns=tvec_feature.get_feature_names())
        word_counts = feature_train.sum(axis=0)
        word_counts.sort_values(ascending = False)
    
        #feature_test = pd.DataFrame(tvec_feature.transform(X_test[feature]).todense(),
        #                            columns=tvec_feature.get_feature_names())
        #logreg = LogisticRegression()
        #logreg.fit(feature_train, y_train)
    
        #y_probs = logreg.predict(feature_test)  
        #target_names = ['below_med', 'above_med']
    
        return word_counts.sort_values(ascending = False), tvec_feature
    
    def top_bottom_wordstfid(feature, good=False):
        """[Collects the most and least used words and phrases]

        Args:
            feature ([col vector]): [features containing string values that can be processed using tfid]
            good (bool, optional): [Allows for setting which end of the current table to build list
            from based on a percentage.]. Defaults to False.

        Returns:
            [list]: [The top or bottom x percent as needed]
        """
        _percent = int(len(tfid_words(feature)[0])*.3)
        if good == True:
            words_out = tfid_words(feature)[0][:_percent]
        else:
            words_out = tfid_words(feature)[0][-_percent:]
        return words_out, tfid_words(feature)[1]
    
    # each of the below is a tuple, the first position going to the binomial
    # feature set, while the 2nd position contains the tfid vector needed for 
    # that feature set.
    top_JobTitles_tfid = top_bottom_wordstfid('JobTitle', good=True)
    bottom_JobTitles_tfid = top_bottom_wordstfid('JobTitle', good=False)
    
    top_Cities_tfid = top_bottom_wordstfid('City', good=True)
    bottom_Cities_tfid = top_bottom_wordstfid('City', good=False)
    
    top_States_tfid = top_bottom_wordstfid('State', good=True)
    bottom_States_tfid = top_bottom_wordstfid('State', good=False)
    
    top_Companies_tfid = top_bottom_wordstfid('Company', good=True)
    bottom_Companies_tfid = top_bottom_wordstfid('Company', good=False)
    
    top_Requirements_tfid = top_bottom_wordstfid('Requirements', good=True)
    bottom_Requirements_tfid = top_bottom_wordstfid('Requirements', good=False)
    
    top_Summaries_tfid = top_bottom_wordstfid('Summary', good=True)
    bottom_Summaries_tfid = top_bottom_wordstfid('Summary', good=False)
    
    top_Descriptions_tfid = top_bottom_wordstfid('Description', good=True)
    bottom_Descriptions_tfid = top_bottom_wordstfid('Description', good=False)
    
    
    # TODO Be able to detect strong correlations, then tune or remove features as need.
    # This may very well be necessary since both sets of new features are drawn from
    # same original features
    # The same tfid feature set is the same between both top_feature/bottom_feature tuples
    
    def tfid_features(split):
        """[Uses the word/phrase lists generated to generate predictive values for each
        table of data (split) under consideration.
        For future analysis, it 'may' help to add ]

        Args:
            split ([DataFrame]): [test train or unseen data]

        Returns:
            [DataFrame]: [Now with tfid features with float values]
        """
        jobTitle_X = pd.DataFrame(top_JobTitles_tfid[1].transform(split["JobTitle"]).todense(),
                                        columns=top_JobTitles_tfid[1].get_feature_names())
        
        City_X = pd.DataFrame(top_Cities_tfid[1].transform(split["City"]).todense(),
                                        columns=top_Cities_tfid[1].get_feature_names())
        
        State_X = pd.DataFrame(top_States_tfid[1].transform(split["State"]).todense(),
                                       columns=top_States_tfid[1].get_feature_names())
        
        Company_X = pd.DataFrame(top_Companies_tfid[1].transform(split["Company"]).todense(),
                                       columns=top_Companies_tfid[1].get_feature_names())
        
        Requirements_X = pd.DataFrame(top_Requirements_tfid[1].transform(split["Requirements"]).todense(),
                                       columns=top_Requirements_tfid[1].get_feature_names())
    
        Summary_X = pd.DataFrame(top_Summaries_tfid[1].transform(split["Summary"]).todense(),
                                       columns=top_Summaries_tfid[1].get_feature_names())
    
        Description_X = pd.DataFrame(top_Descriptions_tfid[1].transform(split["Description"]).todense(),
                                        columns=top_Descriptions_tfid[1].get_feature_names())
        split = pd.concat([split, jobTitle_X, City_X, State_X, Company_X, Requirements_X, Summary_X, Description_X], axis=1)
        return split
    
    def binary_features(split): 
        """[Uses the tfid word lists to provide binary features for each data]

        Args:
            split ([DataFrame]): [test train or unseen data]

        Returns:
            [DataFrame]: [Now with binary features as binary values]
        """
        split["top_JobTitle"] = 0
        for key in top_JobTitles_tfid[0].keys():
            split.loc[(split["JobTitle"].str.contains(f"{key}")), "top_JobTitle"] = 1
        split["bottom_JobTitle"] = 0
        for key in bottom_JobTitles_tfid[0].keys():
            split.loc[(split["JobTitle"].str.contains(f"{key}")), "bottom_JobTitle"] = 1
        split["top_City"] = 0
        for key in top_Cities_tfid[0].keys():
            split.loc[(split["City"].str.contains(f"{key}")), "top_City"] = 1
        split["bottom_City"] = 0
        for key in bottom_Cities_tfid[0].keys():
            split.loc[(split["City"].str.contains(f"{key}")), "bottom_City"] = 1
        split["top_state"] = 0
        for key in top_States_tfid[0].keys():
            split.loc[(split["State"].str.contains(f"{key}")), "top_state"] = 1
        split["bottom_state"] = 0
        for key in bottom_States_tfid[0].keys():
            split.loc[(split["State"].str.contains(f"{key}")), "bottom_state"] = 1
        split["top_company"] = 0
        for key in top_Companies_tfid[0].keys():
            split.loc[(split["Company"].str.contains(f"{key}")), "top_company"] = 1
        split["bottom_compay"] = 0
        for key in bottom_Companies_tfid[0].keys():
            split.loc[(split["Company"].str.contains(f"{key}")), "bottom_compay"] = 1
        split["top_requirements"] = 0
        for key in top_Requirements_tfid[0].keys():
            split.loc[(split["Requirements"].str.contains(f"{key}")), "top_requirements"] = 1
        split["bottom_reqirements"] = 0
        for key in bottom_Requirements_tfid[0].keys():
            split.loc[(split["Requirements"].str.contains(f"{key}")), "bottom_reqirements"] = 1
        split["top_summary"] = 0
        for key in top_Summaries_tfid[0].keys():
            split.loc[(split["Summary"].str.contains(f"{key}")), "top_summary"] = 1
        split["bottom_summary"] = 0
        for key in bottom_Summaries_tfid[0].keys():
            split.loc[(split["Summary"].str.contains(f"{key}")), "bottom_summary"] = 1
        split["top_description"] = 0
        for key in top_Descriptions_tfid[0].keys():
            split.loc[(split["Description"].str.contains(f"{key}")), "top_description"] = 1
        split["bottom_description"] = 0
        for key in bottom_Descriptions_tfid[0].keys():
            split.loc[(split["Description"].str.contains(f"{key}")), "bottom_description"] = 1
        
        # dropping these data columns leaves only the predictive features.
        split.drop(['JobTitle', 'City', 'State', 'Company', 'Requirements', 'Summary', 'Description'], axis=1, inplace=True)
        # TODO get rid of this.
        # Converting 1 and 0 values in matrix to float
        split = split.astype("float")
        return split
    
    # The crux of the rl function is that it contains each of our splits and
    # operates on each in sequence.

    # Featurizing: Train
    X_train = tfid_features(X_train)
    X_train = binary_features(X_train)
    
    # Evaluating: Train
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    #print("Score:", logreg.score(X_train, y_train))
    
    #TODO this needs to be routed to output so webapp can display the predictive
    # strength of our model.
    y_train = y_train.to_numpy().flatten()
    scores = cross_val_score(logreg, X_train, y_train, cv=6)
    #print("Cross-validated scores:", scores)
    
    

    # Featurizing: Test
    X_test = tfid_features(X_test)
    X_test = binary_features(X_test)
    
    # Evaluating: Test
    y_probs_med = logreg.predict(X_test) 

    #print("Score:", logreg.score(X_test, y_test))
    #print(confusion_matrix(y_test_med, y_probs_med,))

    # target_names = ['below', 'above']
    #print(classification_report(y_test_med, y_probs_med, target_names=target_names))
    
    ### Prediction On Unseen Data
    # Much as before, where the target is selected given a string representing the particular
    # quartile we are interested in, except now using a table where the values in our target are null.
    predict_data = data[data[col].isnull()]
    predict_data.reset_index(drop=True, inplace=True)
    
    # Splitting out our predictor variables from the salary data
    predict_X = predict_data[['JobTitle', 'City', 'State', 'Company', 'Requirements', 'Summary', 'Description']]
    predict_X.reset_index(drop=True, inplace=True)
    predict_X.head()
    
    predict_X = tfid_features(predict_X)
    predict_X = binary_features(predict_X)
    
    
    predict_y = logreg.predict(predict_X)
    predict_data[f"{col}_prediction"] = predict_y
    
    #print(len(data[data.median_sal.notnull()]))
    #print(len(predict_data))
    
    # Finally, combing our data that had given salaries, with the data we had
    # to predict salary ranges for.
    # This table is then reused as the rl function is called for each target we're interested in.
    given_sal = data[data[col].notnull()]
    lst = [given_sal, predict_data]
    new_data = pd.concat(lst)
    
    return new_data

def summary_rl():
    """[Calls the rl function to predict salary ranges for each quartile target.
        Using the resultant table's salary predictions and given salaries,
        populates new columns indicating whether or not, predicted or not,
        which bracket each data belongs to.]

    """

    data = pd.read_csv('../data/munged_data_scientist.csv', index_col=0)
    data.reset_index(drop=True, inplace=True)

    # Some tidy that can't seem to happen in Munge...

    data['State'].replace(np.nan, '', regex=True, inplace=True)
    data['City'].replace(np.nan, '', regex=True, inplace=True)

    # EDA: Stats
    # TODO compute and utlize IQR etc in order to stratify analysis further
    # print("salary median: " + str(data["Salary"].median()))
    # print("salary mean: " + str(data["Salary"].mean()))
    # print(data.describe())

    # Creation of targets baseed on our munged data's summary stats.
    count, mean, _, _, q1, q2, q3, _ = data["Salary"].describe()
    data["median_sal"] = np.nan
    data["iqr"] = np.nan
    data["low"] = np.nan
    data["high"] = np.nan

    data.loc[data["Salary"] > data["Salary"].median(), "median_sal"] = 1
    data.loc[data["Salary"] <= data["Salary"].median(), "median_sal"] = 0

    data.loc[(data["Salary"] >= q1) & (data["Salary"] <= q3), "iqr"] = 1
    data.loc[(data["Salary"] <= q1) | (data["Salary"] >= q3), "iqr"] = 0

    data.loc[data["Salary"] <= q1, "low"] = 1
    data.loc[data["Salary"] >= q1, "low"] = 0

    data.loc[data["Salary"] >= q3, "high"] = 1
    data.loc[data["Salary"] <= q3, "high"] = 0

    # 
    col = 'median_sal'
    med = rl(col, data)
    col = 'iqr'
    iqr = rl(col, med)
    col = 'low'
    low = rl(col, iqr)
    col = 'high'
    high = rl(col, low)


    high.to_csv('../data/ml.csv')

