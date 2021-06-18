import csv
from datetime import datetime, date
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from random import random
import time

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")


def get_URL(position, location):
    """[Build a template url]

    Args:
        position ([string]): [job for query]
        location ([string]): [location for query]

    Returns:
        [string]: [formatted url]
    """
    template = 'https://www.indeed.com/jobs?q={}&l={}&fromage=7&sort=date'
    position = position.replace(' ', '+')
    location = location.replace(' ', '+')
    url = template.format(position, location)
    return url


def get_features(web):
    """[Designates desired features and provides for their initial processing]

    Args:
        web ([Data from web pull]): [Single job posting]

    Returns:
        [Data]: [Retieved from pull and processed]
    """
    job_title = web.h2.a.get('title')
    company = web.find('span', 'company').text.strip()
    job_location = web.find('div', 'recJobLoc').get('data-rc-loc')
    post_date = web.find('span', 'date').text
    summary = web.find('div', 'summary').text.strip().replace('\n', ' ')
    today = datetime.today().strftime('%Y-%m-%d')
    job_url = 'https://www.indeed.com' + web.h2.a.get('href')
    
    
    def job_description(job_url):
        """[Retrieves data from job summary page attached to each query result]

        Args:
            job_url ([string]): [url to the specific posting]

        Returns:
            [tuple of strings]: [job requirements, job description]
        """
        # I'd noticed that most Indeed webscrapers either skip the descriptive text contained
        # in the actual posting. Here, I repeat much of the process used to retrieve the job
        # postings but use the url given by those postings to dig a bit deeper.
        response_jobDesc = requests.get(job_url)
        soup = BeautifulSoup(response_jobDesc.text, 'html.parser')
        # https://stackoverflow.com/questions/63231164/indeed-web-scraping-python-selenium-beautifulsoup
        try:
            requirements = soup.find(class_="icl-u-xs-block jobsearch-ReqAndQualSection-item--title").text.replace("\n", "").strip()
        except:
            requirements = 'None'
        try:
            description = soup.find(id="jobDescriptionText").text.replace('\n', '')
        except:
            description = 'None'
        # A nifty little workaround for evading detection.
        time.sleep(3+random()*2)
        return requirements, description
    
    requirements, description = job_description(job_url)

    # this does not exists for all jobs, so handle the exceptions
    salary_tag = web.find('span', 'salaryText')
    if salary_tag:
        salary = salary_tag.text.strip()
    else:
        salary = ''
        
    data = (job_title, company, job_location, post_date, today, summary, salary, job_url, requirements, description)
    return data


def main(position, location):
    """[Conducts the web scraping process]

    Args:
        position ([string]): [job position for indeed.com query]
        position ([string]): [job location for indeed.com query]
        
        Returns:
        [csv]: [scraped data]
    """
    data = []
    url = get_URL(position, location)
    
    # extract the job data
    while True:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        pull = soup.find_all('div', 'jobsearch-SerpJobCard')
        for web in pull:
            datapoint = get_features(web)
            data.append(datapoint)
            # Again, a nifty little workaround for evading detection.
            time.sleep(2+random()*3)
        try:
            url = 'https://www.indeed.com' + soup.find('a', {'aria-label': 'Next'}).get('href')
        except AttributeError:
            break

    name = position.replace(' ','_')
    loc = location.replace(' ','_')
    day = date.today()
    # save the job data
    with open(f'../data/scraped_{name}_{loc}_{day}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['JobTitle', 'Company', 'Location', 'PostDate', 'ExtractDate', 'Summary', 'Pay', 'JobUrl', 'Requirements', 'Description'])
        writer.writerows(data)
