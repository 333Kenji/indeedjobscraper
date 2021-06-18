import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 17

import seaborn as sns
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
sns.set(style="ticks", color_codes=True)


def munge():
    """[Cleaning and Featuring Engineering]

    Args:
        position ([string]): [identifier for scraped dataset]

    Returns:
        [csv]: [data prepared for ML]
    """
    # Using the position variable to select/process scraped data based the on the query that
    # generated it.
    data = pd.read_csv(f'../data/total.csv', index_col=1)
    #TODO just name is salary in scrape.py

    

    def states_(i):
        """[Builds 'State' feature by splitting 'Location']

        Args:
            i ([row of data]): ['Location' value]

        Returns:
            [string]: [state name abbrieviation]
        """
        states = {"AL":"Alabama", "AK":"Alaska", "AS":"American Samoa", "AZ":"Arizona", "AR":"Arkansas",
        "CA":"California", "CO":"Colorado", "CT":"Connecticut", "DE":"Delaware", "DC":"District of Columbia",
        "FL":"Florida", "GA":"Georgia", "GU":"Guam", "HI":"Hawaii", "ID":"Idaho", "IL":"Illinois", "IN":"Indiana",
        "IA":"Iowa", "KS":"Kansas", "KY":"Kentucky", "LA":"Louisiana", "ME":"Maine", "MD":"Maryland", "MA":"Massachusetts",
        "MI":"Michigan", "MN":"Minnesota", "MS":"Mississippi", "MO":"Missouri", "MT":"Montana", "NE":"Nebraska",
        "NV":"Nevada", "NH":"New Hampshire", "NJ":"New Jersey", "NM":"New Mexico", "NY":"New York", "NC":"North Carolina",
        "ND":"North Dakota", "MP":"Northern Mariana Islands", "OH":"Ohio", "OK":"Oklahoma", "OR":"Oregon", "PA":"Pennsylvania",
        "PR":"Puerto Rico", "RI":"Rhode Island", "SC":"South Carolina", "SD":"South Dakota", "TN":"Tennessee",
        "TX":"Texas", "UT":"Utah", "UM":"U.S. Minor Outlying Islands", "VT":"Vermont", "VI":"Virgin Islands", "VA":"Virginia",
        "WA":"Washington", "WV":"West Virginia", "WI":"Wisconsin", "WY":"Wyoming"}
        extras = ["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut", "District ", "of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
        try:
            # These two values are sometimes present instead of a specific state
            if i == 'United States':
                return 'United States'
            elif i == 'Remote':
                return 'Remote'
            elif i == 'New York State':
                return 'NY'
            elif i == 'Washington State':
                return 'WA'
            
            #elif i in extras:
                #return i
            # if neither of the above are all we've been given, we can pull the
            # abbrieviation.
            else:
                for k,v in states.items():
                    if i[-2:] == k:
                        return k
                    elif i == v:
                        return k
        except:
            return i


    def cities_(i):
        """[Builds 'City' feature by splitting 'Location']

        Args:
            i ([row of data]): ['Location' value]

        Returns:
            [string]: [city name from 'Location']
        """
        states = {"AL":"Alabama", "AK":"Alaska", "AS":"American Samoa", "AZ":"Arizona", "AR":"Arkansas",
        "CA":"California", "CO":"Colorado", "CT":"Connecticut", "DE":"Delaware", "DC":"District of Columbia",
        "FL":"Florida", "GA":"Georgia", "GU":"Guam", "HI":"Hawaii", "ID":"Idaho", "IL":"Illinois", "IN":"Indiana",
        "IA":"Iowa", "KS":"Kansas", "KY":"Kentucky", "LA":"Louisiana", "ME":"Maine", "MD":"Maryland", "MA":"Massachusetts",
        "MI":"Michigan", "MN":"Minnesota", "MS":"Mississippi", "MO":"Missouri", "MT":"Montana", "NE":"Nebraska",
        "NV":"Nevada", "NH":"New Hampshire", "NJ":"New Jersey", "NM":"New Mexico", "NY":"New York", "NC":"North Carolina",
        "ND":"North Dakota", "MP":"Northern Mariana Islands", "OH":"Ohio", "OK":"Oklahoma", "OR":"Oregon", "PA":"Pennsylvania",
        "PR":"Puerto Rico", "RI":"Rhode Island", "SC":"South Carolina", "SD":"South Dakota", "TN":"Tennessee",
        "TX":"Texas", "UT":"Utah", "UM":"U.S. Minor Outlying Islands", "VT":"Vermont", "VI":"Virgin Islands", "VA":"Virginia",
        "WA":"Washington", "WV":"West Virginia", "WI":"Wisconsin", "WY":"Wyoming"}

        try:
            # Again, these are sometimes used instead of city, state.
            if i == 'United States':
                return ''
            elif i == 'Remote':
                return 'Remote'
            else:
                city = i[0:-2].strip().replace(',', '')
                return city
        except:
            'None'


    def count_dupes(data):
        """[Small helper for quick data integrity check]

        Args:
            data ([DataFrame]): [scrapped data]

        Returns:
            [tuple]: [number of duplicates (if any), number of unique values]
        """
        dupe = 0
        uniq = 0
        for i in data:
            if i == True:
                dupe += 1
            else:
                uniq +=1
        return dupe, uniq


    def deduper(data):
        """[Dropes duplicates]

        Args:
            data ([DataFrame]): [in munging]

        Returns:
            [DataFrame]: [Now without dupes]
        """
        data = data.drop_duplicates()
        data = data.reset_index(drop=False, inplace=False)
        return data


    def sal_chars(data):
        """[Reduces Pay values to alphanumeric chars only]

        Args:
            data ([DataFrame]): [in munging]

        Returns:
            [DataFrame]: [Now with no special chars]
        """
        data["Pay"] = data["Pay"].str.replace("\n", "")
        data["Pay"] = data["Pay"].str.replace(",", "")
        data["Pay"] = data["Pay"].str.replace("$", "", regex=False)
        return data


    def Pay_period(data):
        """[Builds a column for rate of Pay so a yearly salary can be computed]

        Args:
            data ([DataFrame]): [in munging]

        Returns:
            [DataFrame]: [Now with Pay periods]
        """
        data['Schedule'] = np.nan
        data['Schedule'] = np.where(data['Pay'].str.contains("hour"),"hour",data['Schedule'])
        data['Schedule'] = np.where(data['Pay'].str.contains("week"),"week",data['Schedule'])
        data['Schedule'] = np.where(data['Pay'].str.contains("day"),"day",data['Schedule'])
        data['Schedule'] = np.where(data['Pay'].str.contains("year"),"year",data['Schedule'])
        data['Schedule'] = np.where(data['Pay'].str.contains("NaN"),np.nan,data['Schedule'])
        return data


    def sal_strings(data):
        """[Reduces Pay values to numeric chars only]

        Args:
            data ([DataFrame]): [in munging]

        Returns:
            [DataFrame]: [Now with only numeric chars in Pay col]
        """
        data["Pay"] = data["Pay"].str.replace(" an hour", "")
        data["Pay"] = data["Pay"].str.replace(" a day", "")
        data["Pay"] = data["Pay"].str.replace(" a week", "")
        data["Pay"] = data["Pay"].str.replace(" a month", "")
        data["Pay"] = data["Pay"].str.replace(" a year", "")
        return data


    def split_sal(i):
        """[Converts salaries given as a range to the average of their min/max]

        Args:
            i ([row]): [applied to 'Pay' column]

        Returns:
            [float]: [If given a range, its mean]
        """
        try:
            lst = i.split(' - ',1)
            x = lst[0]
            y = lst[1]
            return (float(x)+float(y))//2
        except:
            return i

    def from_(i):
        #TODO I think this is too simple, find a way to include data that
        # may fall within a range.
        """[If salary is given with a base amount, returns that]

        Args:
            i ([row]): [applied to 'Pay' column]

        Returns:
            [string]: [Lower limit, if given]
        """
        try:
            lst = i.split('From ',1)
            y = lst[1]
            return (y)
        except:
            return i


    def up_to(i):
        # TODO combine this with the above
        """[If salary is given with a max amount, returns that]

        Args:
            i ([row]): [applied to 'Pay' column]

        Returns:
            [string]: [Upper limit, if given]
        """
        try:
            lst = i.split('Up to ',1)
            y = lst[1]
            return (y)
        except:
            return i



    def pDate(row):
        #TODO 64?
        """[Builds a column for date posted. since Indeed.com only gives values for
        postdate relative to day of query.]

        Args:
            i ([row]): [applied to 'PostDate' column]

        Returns:
            [date]: [The actual date the posting was created]
        """
        days_ago = row['PostDate']
        delta = timedelta(days_ago)
        try:
            return row['ExtractDate'] - delta
        except:
            return row


    def annual(data):
        """[Builds an annual salary feature with values for all data]

        Args:
            data ([DataFrame]): [in munging]

        Returns:
            [DataFrame]: [Now with annual salary values]
        """
        data['Salary'] = np.nan
        data['Salary'] = np.where(data['Schedule'].str.contains("hour"), data['Pay']*365/7*40, data['Salary'])
        data['Salary'] = np.where(data['Schedule'].str.contains("day"), data['Pay']*365/7*5, data['Salary'])
        data['Salary'] = np.where(data['Schedule'].str.contains("week"), data['Pay']*365/7, data['Salary'])
        data['Salary'] = np.where(data['Schedule'].str.contains("month"), data['Pay']*365/12, data['Salary'])
        data['Salary'] = np.where(data['Schedule'].str.contains("year"), data['Pay'], data['Salary'])
        return data


    def acronyms(data):
        """[Spells out some commonly encountered acronyms. Supports accuracy of text analysis.]

        Args:
            data ([DataFrame]): [in munging]

        Returns:
            [DataFrame]: [Now with fewer acronymns]
        """
        data["JobTitle"] = data["JobTitle"].str.replace("R&D", "research development")
        data["Summary"] = data["Summary"].str.replace("R&D", "research development")
        data["Description"] = data["Description"].str.replace("R&D", "research development")
        return data


    def chars(data):
        """[summary]

        Args:
            data ([DataFrame]): [in munging]

        Returns:
            [DataFrame]: [Now without special chars]
        """
        cleaning_list = ["+", "$", "/", ",", "?", ".", ";", ":", "-", "@", "!", "&", "%", "^", "*", ")", "(", "\n"]
        for item in cleaning_list:
            data['PostDate'] = data['PostDate'].str.replace(item, " ", regex=False)
            data['Summary'] = data['Summary'].str.replace(item, " ",regex=False)
            data['Description'] = data['Description'].str.replace(item, " ",regex=False)
        return data


    def postD_int(data):
        """[Reduces or converts relative post dates to numeric chars]

        Args:
            data ([DataFrame]): [in munging]

        Returns:
            [DataFrame]: [Now with only numeric values for post date]
        """
        data["PostDate"] = data["PostDate"].str.replace("Active ", "")
        data["PostDate"] = data["PostDate"].str.replace(" day ago", "")
        data["PostDate"] = data["PostDate"].str.replace("%+ days ago", "")
        data["PostDate"] = data["PostDate"].str.replace("+", "")
        data["PostDate"] = data["PostDate"].str.replace(" days ago", "")
        data["PostDate"] = data["PostDate"].str.replace("Just posted", "0")
        data["PostDate"] = data["PostDate"].str.replace("Today", "0")
        data["PostDate"] = data["PostDate"].str.replace("today", "0")
        data['PostDate'] = data['PostDate'].astype('int')
        return data
    


    def roles(data):
        """[Supports web app display by providing website view table with information
        releavent to the job role.]

        Args:
            data ([DataFrame]): [munged]

        Returns:
            [DataFrame]: [Now with specific jobs and roles for each listing]
        """
        #Primary Role
        data['Role'] = ''
        analyst = ['anal']
        eng = ['big data', 'engin', 'data manag', 'data officer']
        ds = ['data scien', 'ml', 'deep', 'model', 'modeler','machine', 'deep', 'ai', 'scientist']


        data['Role'] = np.where(data['Role'].str.contains(''), 'Other', data['Role'])
        for _ in analyst:
            data['Role'] = np.where(data['JobTitle'].str.contains(_), 'data analyst', data['Role'])
        for _ in eng:
            data['Role'] = np.where(data['JobTitle'].str.contains(_), 'data engineer', data['Role'])
        for _ in ds:
            data['Role'] = np.where(data['JobTitle'].str.contains(_), 'data scientist', data['Role'])


        #Focus
        data['Focus'] = ''
        ml = ['ml', 'deep', 'model', 'modeler','machine', 'deep', 'ai']
        sr = ['sr.', 'lead', 'senior', 'manager']
        applied = ['applied']

        for _ in analyst:
            data['Focus'] = np.where(data['JobTitle'].str.contains(_), 'analysis', data['Focus'])
        for _ in ml:
            data['Focus'] = np.where(data['JobTitle'].str.contains(_), 'machine learning', data['Focus'])
        for _ in sr:
            data['Focus'] = np.where(data['JobTitle'].str.contains(_), 'senior', data['Focus'])
        return data

    # Apply the above functions to the selected DataFrame
    # These are in a neceassary order of operation as many functions require some cleaning or
    # featurization to have occured prior to their call/application.
    data["State"] = data["Location"].apply(states_)
    data["City"] = data["Location"].apply(cities_)
    data['ExtractDate']= pd.to_datetime(data['ExtractDate'])
    data = chars(data)
    data = postD_int(data)
    data['DatePosted'] = data.apply( lambda row : pDate(row), axis = 1)
    data = deduper(data)
    data = sal_chars(data)
    data = Pay_period(data)
    data = sal_strings(data)
    data["Pay"] = data["Pay"].apply(split_sal)
    data["Pay"] = data["Pay"].apply(from_)
    data["Pay"] = data["Pay"].apply(up_to)
    data['Pay'] = pd.to_numeric(data['Pay'])
    data = annual(data)
    data = acronyms(data)


    

    # Drop a few cols we no longer need
    data.drop(columns=['Pay','ExtractDate', 'PostDate'], inplace=True)

    for item in ['JobTitle', 'Company', 'Summary', 'Requirements','Description', 'City']:
        data[item] = data[item].str.lower()
    data = roles(data)
    data.to_csv(f'../data/munged_data.csv', index=False)
    return data


