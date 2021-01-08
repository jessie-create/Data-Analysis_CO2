import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

import matplotlib.pyplot as plt
import datetime as datetime


def my_table(label,sheet):
    """
    :param label:define function to label and input the label name
    :param sheet:input the name of sheet
    :return: return a dataset containing specific country GHG information 
    """
    if any(sheet['Country']==label)==True:
        return sheet[sheet['Country']==label]
    else:
        return sheet[sheet['Year']==label]
    
def table_join(table_1,table_2):
    """
    :param table_1:define table_1 which extract from my_table function
    :param table_2:define table_1 which extract from my_table function
    :return: return a dataset with table_1 and table_2 joined 
    """
    result=pd.merge(left=table_1,right=table_2,how='inner',on=['Country','Year'])
    return result

def mean_difference(group_1,group_2):  
    """
    this function is used to test mean difference significance in group1 and group2, they don't need to have 
    same length, they can also contain NaN values. For the mean difference hypothesis, we first need to do F test to test the 
    equal variance. then based the variance we choose different argument in T test to test the mean difference 
    :group_1:it can be a column or a series
    :group_2:a column or a series
    
    """
    group_1=group_1.dropna()
    group_2=group_2.dropna()
    f_stat,f_p=stats.levene(group_1,group_2)
    if f_p<=0.05:
        t_stat,t_p=stats.ttest_ind(group_1,group_2,equal_var=False)
        if t_p<=0.05:
            return print(f'the difference in mean is significant, p={t_p}')
        else:
            return print(f'the difference in mean is not significant, p={t_p}')        
    else:
        t_stat,t_p=stats.ttest_ind(group_1,group_2,equal_var=True)
        if t_p<=0.05:
            return print(f'the difference in mean is significant, p={t_p}')
        else:
            return print(f'the difference in mean is not significant, p={t_p}')  
        
def single_linear(x,y):
    """
    x and y can have NAN values, but after droping the NAN values, their length should be equal
    the function return the coefficient of x, intercept, r square value, p_value of coefficent, and standard error
    """
    x=x.dropna()
    y=y.dropna()
    regression_table=pd.concat([x, y], axis=1, join='inner')
    x=regression_table.iloc[:,0]
    y=regression_table.iloc[:,1]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return print(f'slope={slope}, intercept={intercept}, r_value={r_value}, p_value={p_value}, std_err={std_err}')

def multi_linear(x,y,year):
    """
    :x:it can have NAN values but after droping them x should have same length as y
    :y:it can have NAN values but after droping them x should have same length as x
    :year:it is used to generate time series plot, because the original index is not time, we cannot generate a time series automatically, so we
    need to add this column. it should have the same index as x and y.
    return: a multilinear regression always with constant will be returned. a scatter plot for actaul value and line plot for estimated value will be
    returned
    """
    x=x.dropna(how='any')
    y=y.dropna(how='any')
    index=pd.concat([x, y], axis=1, join='inner').index
    xx=x.loc[index]
    yy=y.loc[index]
    year_s=year.loc[index]
    xx=sm.add_constant(xx)
    my_model=sm.OLS(yy,xx).fit()    
    estimation=my_model.predict()
    plt.scatter(year_s,yy,label='actual value',color='blue')
    plt.plot(year_s,estimation,label='estimated value',color='red')
    plt.legend()
    return print(my_model.summary())and plt.show()

