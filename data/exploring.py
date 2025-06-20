#====================================#
#LIBRARIES
#====================================#
import pandas as pd
import numpy as np
import os
import re 
import matplotlib.pyplot as plt



working_path = '/Users/justynakubot/Desktop/EDU/Hackaton/'
file_name = 'adult_sal.csv'
# If it needed set working dir 
os.chdir(working_path)

df = pd.read_csv(file_name) # to change if format different than csv


number_of_na = df.isnull().sum() # pd series
cols_by_dtypes = df.dtypes.groupby(df.dtypes).groups # dictionary 
cols_by_dtypes

df.describe()
# CHECK NAS

def explore_the_df(df):
    # Local Variables 
    nrow,ncol = get_size(df)
    # Check nas 
    percent_of_nas = check_nas(df)
    # Check coltypes 
    col_types = get_col_types(df)
    # Explore the values 
    summaries_strings(df,col_types)
      
    # Summary 

    return(df,nrow,ncol,percent_of_nas,col_types)

def get_size(df):

    # Check how massive the data is, ho
    nrow = df.shape[0] # define number of rows
    ncol = df.shape[1] # define number of columns 
    print(f'In data set you can find {ncol} columns and {nrow} rows')
    return(nrow,ncol)


def summaries_strings(df,col_types):

    if len(col_types['object'])==0:
        print('No String Values to check')
    else:
        print('Exploring String Columns')

        plot_string_summary(df[col_types['object']])
        df_with_others = change_label_of_rare(df[col_types['object']])
        plot_string_summary(df_with_others)

    

def change_label_of_rare(df):
    df = df.apply(lambda col: col.str.upper())

    # 
    for col in df.columns.to_list():
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < 0.05].index
        df[col] = df[col].apply(lambda x: 'OTHERS' if x in rare else x)
    return(df)

    
def plot_string_summary(df):
    num_cols = len(df.columns.to_list())
    fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(10, 4 * num_cols))

    if num_cols == 1:
        axes = [axes]  

    for ax, col in zip(axes, df.columns.to_list()):
        df[col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f"Frequency of different values: {col}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    #Display plots
    plt.tight_layout()
    plt.show()
    

def summaries_intigers(df):

    
    if len(df.select_dtypes(include=['number']).columns.to_list())==0:
        print('No String Values to check')
    else:
        print("Exploring Numeric Columns")

        df_num = df.select_dtypes(include=['number'])
        summary_of_df = df_num.describe().T
        print(f'Summary statistics of numeric columns')
        print(summary_of_df)

        intiger_histograms(df_num)
        cor_matrix(df_num)
        box_plots(df_num)
        
    return()

def intiger_histograms(df):
    df.hist(bins=30, figsize=(15, 10))
    plt.suptitle("Histograms of numerics")
    plt.tight_layout()
    plt.show()

def cor_matrix(df):

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap='coolwarm')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    fig.colorbar(cax)
    plt.title("Correlation", y=1.15)
    plt.show()

def box_plots(df):
    for col in df.columns:
        plt.figure(figsize=(6, 2))
        plt.boxplot(df[col].dropna(), vert=False)
        plt.title(f"Boxplot: {col}")
        plt.xlabel(col)
        plt.show()



def check_nas(df):
    if ~( df.isnull().values.any()):
        print('No obviously missing values')
    else:
        percent_of_nas = 100 * df.isnull().sum() / len(df)
        percent_of_nas = percent_of_nas[percent_of_nas>0]
        print('Percentage of missing values')
        print(percent_of_nas)
        return(percent_of_nas)


def get_col_types(df):

    column_types = {
    str(dtype): list(columns)
    for dtype, columns in df.dtypes.groupby(df.dtypes).groups.items()
    }
    for i in column_types.keys():
        print(f'Data set contains following {i} columns')
        print(column_types[i])
    return(column_types)




# DEAL WITH NAS

# CHECK COLUMNS

# GROUP COLUMNS

# TRANSFORM STRINGS AND LOOK FOR UNIQUE VALUES

# LOOK FOR DUPLICATES

# TRANSFORM RARE VALUES INTO OTHERS

# 



df = pd.read_csv(file_name)

df.head()

df.isna().sum()



df.info()

df.dtypes

### GRUPOWANIE KOLUMN

#====================================#
#PARAMETERS AND PATHS
#====================================#
do_save = False # Whenever you want to save outputs of the analysis 

#====================================#
#LOAD INPUTS
#====================================#
def input_load(file_name):

    
    
    




    return(df)
    
#====================================#
#MANAGE
#====================================#


#====================================#
#MODELING FUNCTIONS
#====================================#

#====================================#
#OUTPUTS
#====================================#



#====================================#
#FUNCTIONS USED TO PERFORM EACH COMPONENTS
#====================================#





#====================================#
#LIBRARIES DATA EXPLORING
#====================================#


#====================================#
#LIBRARIES INPUT PROCESING
#====================================#



#====================================#
#LIBRARIES MODEL FITING
#====================================#
