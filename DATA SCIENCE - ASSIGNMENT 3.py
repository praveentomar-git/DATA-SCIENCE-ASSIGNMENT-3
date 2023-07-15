#!/usr/bin/env python
# coding: utf-8

# 1. Scenario: A company wants to analyze the sales performance of its products in different regions. They have collected the following data:
#    Region A: [10, 15, 12, 8, 14]
#    Region B: [18, 20, 16, 22, 25]
#    Calculate the mean sales for each region.

# In[1]:


import statistics as stat
A=[10, 15, 12, 8, 14]
B=[18, 20, 16, 22, 25]
print("Mean Sales of Region A :", stat.mean(A))
print("Mean Sales of Region B :", stat.mean(B))


# 2. Scenario: A survey is conducted to measure customer satisfaction on a scale of 1 to 5. The data collected is as follows:
#    [4, 5, 2, 3, 5, 4, 3, 2, 4, 5]
#    Calculate the mode of the survey responses.

# In[2]:


customer_satisfaction= [4, 5, 2, 3, 5, 4, 3, 2, 4, 5]
print("Mode :", stat.mode(customer_satisfaction))
print("For Multi_Modal :", stat.multimode(customer_satisfaction))


# 3. Scenario: A company wants to compare the salaries of two departments. The salary data for Department A and Department B are as follows:
#    Department A: [5000, 6000, 5500, 7000]
#    Department B: [4500, 5500, 5800, 6000, 5200]
#    Calculate the median salary for each department.

# In[3]:


A=[5000, 6000, 5500, 7000]
B=[4500, 5500, 5800, 6000, 5200]
print("Median Salary of Department A :", stat.median(A))
print("Median Salary of Department B :", stat.median(B))


# 4. Scenario: A data analyst wants to determine the variability in the daily stock prices of a company. The data collected is as follows:
#    [25.5, 24.8, 26.1, 25.3, 24.9]
#    Calculate the range of the stock prices.

# In[4]:


data=[25.5, 24.8, 26.1, 25.3, 24.9]
print("Range of the stock price :", round(max(data)-min(data),2))


# In[5]:


import numpy as np
np.ptp(data)        #ptp stands here peek to peek means range


# 5. Scenario: A study is conducted to compare the performance of two different teaching methods. The test scores of the students in each group are as follows:
#    Group A: [85, 90, 92, 88, 91]
#    Group B: [82, 88, 90, 86, 87]
#    Perform a t-test to determine if there is a significant difference in the mean scores between the two groups.

# In[6]:


Group_A=[85, 90, 92, 88, 91] 
Group_B=[82, 88, 90, 86, 87]
from scipy import stats
t,p=stats.ttest_ind(Group_A,Group_B)
print("t-statistic :", t)
print("p-value :", p)


# The t-statistic represents the difference in means between the two groups and the p-value indicates the statistical significance of the difference.

# p-value>0.05(significance level), we fail to reject the null hypothesis of no difference between the groups and conclude that there is no significant difference in the mean scores

# 6. Scenario: A company wants to analyze the relationship between advertising expenditure and sales. The data collected is as follows:
#    Advertising Expenditure (in thousands): [10, 15, 12, 8, 14]
#    Sales (in thousands): [25, 30, 28, 20, 26]
#    Calculate the correlation coefficient between advertising expenditure and sales.

# In[7]:


expenditure=[10, 15, 12, 8, 14] 
sales=[25, 30, 28, 20, 26] 
np.corrcoef(expenditure, sales)[0,1]
#We use [0, 1] to access the correlation coefficient value from the resulting correlation matrix


# In[8]:


import seaborn as sns
sns.regplot(x=expenditure,y=sales)


# 7. Scenario: A survey is conducted to measure the heights of a group of people. The data collected is as follows:
#    [160, 170, 165, 155, 175, 180, 170]
#    Calculate the standard deviation of the heights.

# In[9]:


height=[160, 170, 165, 155, 175, 180, 170]
stdev=stat.stdev(height)
print("standard deviation of the heights :", stdev)


# 
# 8. Scenario: A company wants to analyze the relationship between employee tenure and job satisfaction. The data collected is as follows:
#    Employee Tenure (in years): [2, 3, 5, 4, 6, 2, 4]
#    Job Satisfaction (on a scale of 1 to 10): [7, 8, 6, 9, 5, 7, 6]
#    Perform a linear regression analysis to predict job satisfaction based on employee tenure.

# In[10]:


employee_tenure=[2, 3, 5, 4, 6, 2, 4] 
job_satisfaction=[7, 8, 6, 9, 5, 7, 6] 
m,c,r,p,s_error=stats.linregress(employee_tenure,job_satisfaction)
print("slope :",m) # The slope represents the rate of change in job satisfaction with respect to employee tenure.
print("intercept :",c) #The intercept is the estimated job satisfaction when employee tenure is zero.
print("r_squared :", r**2)  # r-value is correlation coefficient and r_squared is the proprtion of the variance in job satisfaction that can be explained by employee tenure.
print("p-value :", p)  #The p-value assesses the significance of the linear regression relationship. 
print("standard_error :", s_error) #The standard error measures the variability in the predicted job satisfaction values.
sns.regplot(x=employee_tenure,y=job_satisfaction)


# 9. Scenario: A study is conducted to compare the effectiveness of two different medications. The recovery times of the patients in each group are as follows:
#    Medication A: [10, 12, 14, 11, 13]
#    Medication B: [15, 17, 16, 14, 18]
#    Perform an analysis of variance (ANOVA) to determine if there is a significant difference in the mean recovery times between the two medications.

# In[11]:


medication_a = [10, 12, 14, 11, 13]
medication_b = [15, 17, 16, 14, 18]
f, p = stats.f_oneway(medication_a, medication_b)
print("f-statistic:", f)
print("p-value:", p)


# F-statistic measures the ratio of between-group variability to within-group variability. Higher F-statistic values indicate larger differences between the groups compared to the variability within each group.

# P-value < 0.05 (significance level) suggests strong evidence to reject the null hypothesis and conclude that there is a significant difference in the mean recovery times between the two medications.

# 
# 10. Scenario: A company wants to analyze customer feedback ratings on a scale of 1 to 10. The data collected is
# 
#  as follows:
#     [8, 9, 7, 6, 8, 10, 9, 8, 7, 8]
#     Calculate the 75th percentile of the feedback ratings.

# In[12]:


feedback_rating= [8, 9, 7, 6, 8, 10, 9, 8, 7, 8]
np.quantile(feedback_rating, 0.75)


# 11. Scenario: A quality control department wants to test the weight consistency of a product. The weights of a sample of products are as follows:
#     [10.2, 9.8, 10.0, 10.5, 10.3, 10.1]
#     Perform a hypothesis test to determine if the mean weight differs significantly from 10 grams.

# In[13]:


weight_of_samples=[10.2, 9.8, 10.0, 10.5, 10.3, 10.1] 
pop_mean=10
t,p_value=stats.ttest_1samp(weight_of_samples,pop_mean)
print('t-statistics :',t)
print('p-value :', p_value)

alpha = 0.05

if p_value < alpha:
    print("The mean weight differs significantly from 10 grams (reject H0)")
else:
    print("The mean weight does not differ significantly from 10 grams (fail to reject H0)")


# 12. Scenario: A company wants to analyze the click-through rates of two different website designs. The number of clicks for each design is as follows:
#     Design A: [100, 120, 110, 90, 95]
#     Design B: [80, 85, 90, 95, 100]
#     Perform a chi-square test to determine if there is a significant difference in the click-through rates between the two designs.

# In[14]:


design_a = [100, 120, 110, 90, 95]
design_b = [80, 85, 90, 95, 100]

observed_freq = np.array([design_a, design_b])

observed_freq_flat = observed_freq.flatten()

print(observed_freq_flat)

chi2_statistic, p_value = stats.chisquare(observed_freq_flat)

print(chi2_statistic)
print(p_value)

alpha = 0.05

if p_value < alpha:
    print("There is a significant difference in the click-through rates between the two designs (reject H0)")
else:
    print("There is no significant difference in the click-through rates between the two designs (fail to reject H0)")


# 13. Scenario: A survey is conducted to measure customer satisfaction with a product on a scale of 1 to 10. The data collected is as follows:
#     [7, 9, 6, 8, 10, 7, 8, 9, 7, 8]
#     Calculate the 95% confidence interval for the population mean satisfaction score.

# In[15]:


import statsmodels.stats.api as sms

sample_data =  [7, 9, 6, 8, 10, 7, 8, 9, 7, 8]

confidence_interval = sms.DescrStatsW(sample_data).tconfint_mean()

print(f"The 95% confidence interval for the population mean is: {confidence_interval}")


# In[16]:


satisfaction_scores = [7, 9, 6, 8, 10, 7, 8, 9, 7, 8]

sample_mean = np.mean(satisfaction_scores)
sample_std = np.std(satisfaction_scores, ddof=1)

# Set the confidence level (1 - alpha)
confidence_level = 0.95

# Calculate the critical value (two-tailed t-distribution)
critical_value = stats.t.ppf((1 + confidence_level) / 2, df=len(satisfaction_scores) - 1)

# Calculate the margin of error
margin_of_error = critical_value * (sample_std / np.sqrt(len(satisfaction_scores)))

# Calculate the confidence interval
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

# Print the confidence interval
print(f"The 95% confidence interval for the population mean satisfaction score is: {confidence_interval}")


# 14. Scenario: A company wants to analyze the effect of temperature on product performance. The data collected is as follows:
#     Temperature (in degrees Celsius): [20, 22, 23, 19, 21]
#     Performance (on a scale of 1 to 10): [8, 7, 9, 6, 8]                                                                           Perform a simple linear regression to predict performance based on temperature.

# In[17]:


temperature=[20, 22, 23, 19, 21]
performance=[8, 7, 9, 6, 8]
m,c,r,p,s_error=stats.linregress(temperature,performance)
print("slope :",m) 
print("intercept :",c) 
print("r_squared :", r**2)  
print("p-value :", p)  
print("standard_error :", s_error) 
sns.regplot(x=temperature,y=performance)


# 15. Scenario: A study is conducted to compare the preferences of two groups of participants. The preferences are measured on a Likert scale from 1 to 5. The data collected is as follows:
#     Group A: [4, 3, 5, 2, 4]
#     Group B: [3, 2, 4, 3, 3]
#     Perform a Mann-Whitney U test to determine if there is a significant difference in the median preferences between the two groups.
# 

# In[18]:


group_a = [4, 3, 5, 2, 4]
group_b = [3, 2, 4, 3, 3]

statistic, p_value = stats.mannwhitneyu(group_a, group_b)

alpha = 0.05

if p_value < alpha:
    print("There is a significant difference in the median preferences between the two groups (reject H0)")
else:
    print("There is no significant difference in the median preferences between the two groups (fail to reject H0)")


# 
# 16. Scenario: A company wants to analyze the distribution of customer ages. The data collected is as follows:
#     [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
#     Calculate the interquartile range (IQR) of the ages.

# In[19]:


data=[25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
iqr=np.quantile(data,0.75)-np.quantile(data,0.25)
print(iqr)


# In[20]:


stats.iqr(data)


#            17. Scenario: A study is conducted to compare the performance of three different machine learning algorithms. The accuracy scores for each algorithm are as follows:
#     Algorithm A: [0.85, 0.80, 0.82, 0.87, 0.83]
#     Algorithm B: [0.78, 0.82, 0.84, 0.80, 0.79]
#     Algorithm C: [0.90, 0.88, 0.89, 0.86, 0.87]
#     Perform a Kruskal-Wallis test to determine if there is a significant difference in the median accuracy scores between the algorithms.

# In[21]:


Algorithm_A= [0.85, 0.80, 0.82, 0.87, 0.83]
Algorithm_B= [0.78, 0.82, 0.84, 0.80, 0.79]
Algorithm_C= [0.90, 0.88, 0.89, 0.86, 0.87]
statistic, p_value=stats.kruskal(Algorithm_A,Algorithm_B, Algorithm_C)
print('p-value :',p_value)
alpha = 0.05
if p_value < alpha:
    print("There is a significant difference in the median preferences between the two groups (reject H0)")
else:
    print("There is no significant difference in the median preferences between the two groups (fail to reject H0)")


# 
# 18. Scenario: A company wants to analyze the effect of price on sales. The data collected is as follows:
#     Price (in dollars): [10, 15, 12, 8, 14]
#     Sales: [100, 80, 90, 110, 95]
#     Perform a simple linear regression to predict
# 
#  sales based on price.

# In[22]:


price= [10, 15, 12, 8, 14]
sales=[100, 80, 90, 110, 95] 
m,c,r,p,s_error=stats.linregress(price,sales)
print("slope :",m) 
print("intercept :",c) 
print("r_squared :", r**2)  
print("p-value :", p)  
print("standard_error :", s_error) 
sns.regplot(x=price,y=sales)


# 19. Scenario: A survey is conducted to measure the satisfaction levels of customers with a new product. The data collected is as follows:
#     [7, 8, 9, 6, 8, 7, 9, 7, 8, 7]
#     Calculate the standard error of the mean satisfaction score.

# In[23]:


satisfaction_scores = [7, 8, 9, 6, 8, 7, 9, 7, 8, 7]

standard_deviation = stat.stdev(satisfaction_scores)

sample_size = len(satisfaction_scores)

#standard error 
standard_error = standard_deviation / np.sqrt(sample_size)

print(f"The standard error of the mean satisfaction score is: {standard_error}")


# 20. Scenario: A company wants to analyze the relationship between advertising expenditure and sales. The data collected is as follows:
#     Advertising Expenditure (in thousands): [10, 15, 12, 8, 14]
#     Sales (in thousands): [25, 30, 28, 20, 26]
#     Perform a multiple regression analysis to predict sales based on advertising expenditure.

# In[24]:


import statsmodels.api as sm
advertising_expenditure = [10, 15, 12, 8, 14]
sales = [25, 30, 28, 20, 26]

# Add a constant term to the independent variable
expenditure_with_constant = sm.add_constant(advertising_expenditure)

# Perform multiple regression analysis
model = sm.OLS(sales, expenditure_with_constant)
results = model.fit()

# Print the regression results
print(results.summary())

