#====================================#
#LIBRARIES
#====================================#
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import config

EDA_DIR = Path(config.settings.plot_dir) / "eda"


def normalize_categorical(df, min_freq):
    
    df_result = df.copy()

    # Wybór kolumn tekstowych
    string_cols = df_result.select_dtypes(include=['object', 'string']).columns

    for col in string_cols:
        # Zamiana na wielkie litery
        df_result[col] = df_result[col].str.upper()

        # Częstość wartości w kolumnie
        value_counts = df_result[col].value_counts(normalize=True)

        # Wartości poniżej progu częstości
        rare_values = value_counts[value_counts < min_freq].index

        # Zamiana rzadkich na 'OTHERS'
        df_result[col] = df_result[col].apply(lambda x: 'OTHERS' if x in rare_values else x)

    return df_result


def split_by_missing(df, custom_na):
    # Zamiana niestandardowych wartości na np.NaN
    df_clean = df.replace(custom_na, pd.NA)
    
    # Wiersze niekompletne (z co najmniej jednym NA)
    incomplete_rows = df_clean[df_clean.isna().any(axis=1)]

    # Wiersze kompletne (bez NA)
    complete_rows = df_clean[df_clean.notna().all(axis=1)]

    return incomplete_rows, complete_rows


def handle_missing_values(df,method = 'mean'):
    df_filled = df.copy()

    if method == 'mean':
        for col in df_filled.select_dtypes(include='number').columns:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())

    elif method == 'median':
        for col in df_filled.select_dtypes(include='number').columns:
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())

    elif method == 'mode':
        for col in df_filled.columns:
            mode_val = df_filled[col].mode(dropna=True)
            if not mode_val.empty:
                df_filled[col] = df_filled[col].fillna(mode_val[0])

    elif method == 'ffill':
        df_filled = df_filled.fillna(method='ffill')

    elif method == 'bfill':
        df_filled = df_filled.fillna(method='bfill')

    elif method == 'drop':
        df_filled = df_filled.dropna()

    else:
        raise ValueError(f"Nieznana metoda: {method}. Wybierz z: 'mean', 'median', 'mode', 'ffill', 'bfill', 'drop'.")

    return df_filled


def transfrom_the_df(df,na_cods = None,method='drop',min_freq=.01):

    # standarazie nas
    if na_cods!=None:
        incomplete_rows, complete_rows = split_by_missing(df,na_cods)
        incomplete_rows = handle_missing_values(incomplete_rows,method)
        df = pd.concat([complete_rows, incomplete_rows])

    df = normalize_categorical(df,min_freq)

    return df


def summaries_strings(df,col_types):

    if col_types:
        print('No String Values to check')
    else:
        print('Exploring String Columns')

        plot_string_summary(df[col_types['object']])
        df_with_others = change_label_of_rare(df[col_types['object']])
        plot_string_summary(df_with_others)


def get_col_types(df):

    column_types = {
    str(dtype): list(columns)
    for dtype, columns in df.dtypes.groupby(df.dtypes).groups.items()
    }
    for i in column_types.keys():
        print(f'Data set contains following {i} columns')
        print(column_types[i])
    return(column_types)


def check_nas(df):
    if ~( df.isnull().values.any()):
        print('No obviously missing values')
    else:
        percent_of_nas = 100 * df.isnull().sum() / len(df)
        percent_of_nas = percent_of_nas[percent_of_nas>0]
        print('Percentage of missing values')
        print(percent_of_nas)
        return(percent_of_nas)


def get_size(df):

    # Check how massive the data is, ho
    nrow = df.shape[0] # define number of rows
    ncol = df.shape[1] # define number of columns 
    print(f'In data set you can find {ncol} columns and {nrow} rows')
    return(nrow,ncol)
    

def intiger_histograms(df, out_path: Path = EDA_DIR / "histograms.png"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.hist(bins=30, figsize=(15, 10))
    plt.suptitle("Histograms of numerics")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def cor_matrix(df, out_path: Path = EDA_DIR / "correlation_matrix.png"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    corr = df.corr()
    corr.to_csv(out_path.with_suffix(".csv"))
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap="coolwarm")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    fig.colorbar(cax)
    plt.title("Correlation", y=1.15)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def box_plots(df, out_dir: Path = EDA_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for col in df.columns:
        path = out_dir / f"boxplot_{col}.png"
        plt.figure(figsize=(6, 2))
        plt.boxplot(df[col].dropna(), vert=False)
        plt.title(f"Boxplot: {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        paths.append(path)
    return paths


def summaries_intigers(df, out_dir: Path = EDA_DIR):
    if len(df.select_dtypes(include=["number"]).columns.to_list()) == 0:
        print("No String Values to check")
        return None

    print("Exploring Numeric Columns")

    out_dir.mkdir(parents=True, exist_ok=True)
    df_num = df.select_dtypes(include=["number"])
    summary_of_df = df_num.describe().T
    print("Summary statistics of numeric columns")
    print(summary_of_df)

    summary_path = out_dir / "numeric_summary.csv"
    summary_of_df.to_csv(summary_path)

    intiger_histograms(df_num, out_dir / "histograms.png")
    cor_matrix(df_num, out_dir / "correlation_matrix.png")
    box_plots(df_num, out_dir)

    return summary_of_df


def explore_df_target(df, target_col, out_dir: Path = EDA_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    predictors = [c for c in df.columns if c != target_col]

    for col in predictors:
        print(f"Predictor vs {col}")

        if pd.api.types.is_numeric_dtype(df[col]):
            stats = df.groupby(target_col)[col].describe()
            stats_path = out_dir / f"{col}_by_{target_col}.csv"
            stats.to_csv(stats_path)

            df.boxplot(column=col, by=target_col, grid=False, figsize=(6, 4))
            plt.title(f"{col} vs {target_col}")
            plt.xlabel(target_col)
            plt.ylabel(col)
            plt.tight_layout()
            plt.savefig(out_dir / f"{col}_by_{target_col}.png")
            plt.close()

        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            contingency = pd.crosstab(df[col], df[target_col], normalize="index")
            contingency.round(2).to_csv(out_dir / f"{col}_by_{target_col}.csv")

            contingency.plot(kind="bar", stacked=True, figsize=(6, 4))
            plt.title(f"{col} vs {target_col}")
            plt.ylabel("Percent")
            plt.xlabel(col)
            plt.legend(title=target_col)
            plt.tight_layout()
            plt.savefig(out_dir / f"{col}_by_{target_col}.png")
            plt.close()


def explore_the_df(df, target, out_dir: Path = EDA_DIR):
    
    # Local Variables 
    nrow,ncol = get_size(df)
    # Check nas 
    percent_of_nas = check_nas(df)
    # Check coltypes 
    col_types = get_col_types(df)
    # Explore the values
    summaries_strings(df,col_types)

    summaries_intigers(df, out_dir)

    # Final explore 
    explore_df_target(df, target, out_dir)

      
    # Summary 

    return df, nrow, ncol, percent_of_nas, col_types



def change_label_of_rare(df):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.upper()
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < 0.05].index
        df[col] = df[col].apply(lambda x: 'OTHERS' if x in rare else x)
    return df

    
# def plot_string_summary(df):
#     num_cols = len(df.columns.to_list())
#     fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(10, 4 * num_cols))

#     if num_cols == 1:
#         axes = [axes]  

#     for ax, col in zip(axes, df.columns.to_list()):
#         df[col].value_counts().plot(kind='bar', ax=ax)
#         ax.set_title(f"Frequency of different values: {col}")
#         ax.set_xlabel("Value")
#         ax.set_ylabel("Count")
#     #Display plots
#     plt.tight_layout()
#     plt.show()
    
# plot_string_summary(df) # this to be saved


def eksplore_regression(df, target_col, out_dir: Path = EDA_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    predictors = [col for col in df.columns if col != target_col]

    for col in predictors:
        print(f"\n📊 Variable: {col}")

        if pd.api.types.is_numeric_dtype(df[col]):
            corr = df[[col, target_col]].corr().iloc[0, 1]
            print(f"Correlation: {round(corr, 3)}")

            plt.figure(figsize=(5, 4))
            plt.scatter(df[col], df[target_col], alpha=0.6)
            plt.title(f"{col} vs {target_col}\nCorrelation: {round(corr, 3)}")
            plt.xlabel(col)
            plt.ylabel(target_col)
            plt.tight_layout()
            plt.savefig(out_dir / f"scatter_{col}_vs_{target_col}.png")
            plt.close()

        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            desc = df.groupby(col)[target_col].describe()
            desc.to_csv(out_dir / f"{col}_stats.csv")

            df.groupby(col)[target_col].mean().sort_values().plot(kind="bar", figsize=(6, 4))
            plt.title(f"Average {target_col} by {col}")
            plt.ylabel(f"Average {target_col}")
            plt.xlabel(col)
            plt.tight_layout()
            plt.savefig(out_dir / f"avg_{target_col}_by_{col}.png")
            plt.close()

