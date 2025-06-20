#====================================#
#LIBRARIES
#====================================#
from pathlib import Path
from typing import Dict, Tuple, Any

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt



def run_data_exploration(
    df: pd.DataFrame,
    target: str,
    *,
    out_dir: str | Path | None = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Run all EDA routines and save results.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data including the target column.
    target : str
        Name of the target column.
    out_dir : str or Path, optional
        Directory where plots will be saved. Defaults to ``plots/eda``.

    Returns
    -------
    (dict, dict)
        Mapping of collected statistics and mapping of plot titles to file
        paths.
    """

    out = Path(out_dir) if out_dir is not None else Path("plots/eda")
    out.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {}
    plots: Dict[str, str] = {}

    nrow, ncol = get_size(df)
    stats["dataset shape"] = pd.Series({"rows": nrow, "cols": ncol})

    percent_of_nas = check_nas(df)
    if percent_of_nas is not None:
        stats["missing values"] = percent_of_nas

    col_types = get_col_types(df)
    stats["column types"] = col_types

    str_stats, str_plots = summaries_strings(df, col_types, out_dir=out)
    for name, val in str_stats.items():
        stats[name] = val
    plots.update(str_plots)

    num_stats, num_plots = summaries_intigers(df, out_dir=out)
    if num_stats is not None:
        stats["numeric summary"] = num_stats
    plots.update(num_plots)

    target_stats, target_plots = explore_df_target(df, target, out_dir=out)
    for name, val in target_stats.items():
        stats[name] = val
    plots.update(target_plots)

    return stats, plots

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

def transfrom_the_df(df,na_cods = None,method='drop',min_freq=.01):

    # standarazie nas
    if na_cods!=None:
        incomplete_rows, complete_rows = split_by_missing(df,na_cods)
        incomplete_rows = handle_missing_values(incomplete_rows,method)
        df = pd.concat([complete_rows, incomplete_rows])

    df = normalize_categorical(df,min_freq)

    return(df)

  
def explore_the_df(df,target):
    
    # Local Variables 
    nrow,ncol = get_size(df)
    # Check nas 
    percent_of_nas = check_nas(df)
    # Check coltypes 
    col_types = get_col_types(df)
    # Explore the values 
    summaries_strings(df,col_types)
    summaries_intigers(df)

    # Final explore 
    explore_df_target(df,target)

      
    # Summary 

    return(df,nrow,ncol,percent_of_nas,col_types)

def get_size(df):

    # Check how massive the data is, ho
    nrow = df.shape[0] # define number of rows
    ncol = df.shape[1] # define number of columns 
    print(f'In data set you can find {ncol} columns and {nrow} rows')
    return(nrow,ncol)

def summaries_strings(
    df: pd.DataFrame,
    col_types: Dict[str, list],
    *,
    out_dir: Path | None = None,
) -> Tuple[Dict[str, pd.Series], Dict[str, str]]:
    """Summaries and bar plots for string features.

    Returns dictionaries of value counts and saved plot paths.
    """

    string_columns = (
        col_types.get("object", [])
        + col_types.get("string", [])
        + col_types.get("category", [])
    )

    if len(string_columns) == 0:
        print("No String Values to check")
        return {}, {}

    print("Exploring String Columns")
    counts = {col: df[col].value_counts() for col in string_columns}
    plot_paths = plot_string_summary(df[string_columns], out_dir=out_dir)

    df_with_others = change_label_of_rare(df[string_columns])
    plot_paths.update(
        {
            f"{col}_others": p
            for col, p in plot_string_summary(df_with_others, out_dir=out_dir).items()
        }
    )

    return counts, plot_paths

    

def change_label_of_rare(df):
    df = df.apply(lambda col: col.str.upper())

    # 
    for col in df.columns.to_list():
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < 0.05].index
        df[col] = df[col].apply(lambda x: 'OTHERS' if x in rare else x)
    return(df)

    
def plot_string_summary(df: pd.DataFrame, *, out_dir: Path | None = None) -> Dict[str, str]:
    """Create bar plots for string columns and optionally save them."""

    plot_paths: Dict[str, str] = {}
    for col in df.columns.to_list():
        plt.figure(figsize=(6, 4))
        df[col].value_counts().plot(kind="bar")
        plt.title(f"Frequency of different values: {col}")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.tight_layout()
        if out_dir is not None:
            path = Path(out_dir) / f"string_{col}.png"
            plt.savefig(path)
            plot_paths[f"string_{col}"] = str(path)
            plt.close()
        else:
            plt.show()
    return plot_paths
    

def summaries_intigers(df: pd.DataFrame, *, out_dir: Path | None = None) -> Tuple[pd.DataFrame | None, Dict[str, str]]:
    """Summarise numeric columns and generate diagnostic plots."""

    if len(df.select_dtypes(include=["number"]).columns.to_list()) == 0:
        print("No String Values to check")
        return None, {}

    print("Exploring Numeric Columns")

    df_num = df.select_dtypes(include=["number"])
    summary_of_df = df_num.describe().T
    print("Summary statistics of numeric columns")
    print(summary_of_df)

    plot_paths: Dict[str, str] = {}
    plot_paths.update(intiger_histograms(df_num, out_path=Path(out_dir) / "numeric_hist.png" if out_dir else None))
    plot_paths.update(cor_matrix(df_num, out_path=Path(out_dir) / "correlation.png" if out_dir else None))
    plot_paths.update(box_plots(df_num, out_dir=out_dir))

    return summary_of_df, plot_paths

def intiger_histograms(df: pd.DataFrame, *, out_path: Path | None = None) -> Dict[str, str]:
    df.hist(bins=30, figsize=(15, 10))
    plt.suptitle("Histograms of numerics")
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
        plt.close()
        return {"numeric_histograms": str(out_path)}
    else:
        plt.show()
        return {}

def cor_matrix(df: pd.DataFrame, *, out_path: Path | None = None) -> Dict[str, str]:
    corr = df.corr()
    print(corr)
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap="coolwarm")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    fig.colorbar(cax)
    plt.title("Correlation", y=1.15)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
        plt.close()
        return {"correlation": str(out_path)}
    else:
        plt.show()
        return {}

def box_plots(df: pd.DataFrame, *, out_dir: Path | None = None) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for col in df.columns:
        plt.figure(figsize=(6, 2))
        plt.boxplot(df[col].dropna(), vert=False)
        plt.title(f"Boxplot: {col}")
        plt.xlabel(col)
        plt.tight_layout()
        if out_dir is not None:
            path = Path(out_dir) / f"box_{col}.png"
            plt.savefig(path)
            paths[f"box_{col}"] = str(path)
            plt.close()
        else:
            plt.show()
    return paths

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

def explore_df_target(
    df: pd.DataFrame,
    target_col: str,
    *,
    out_dir: Path | None = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Explore relation between predictors and the target."""

    predictors = [col for col in df.columns if col != target_col]
    stats: Dict[str, pd.DataFrame] = {}
    paths: Dict[str, str] = {}

    for col in predictors:
        print(f"Predictor vs {col}")

        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df.groupby(target_col)[col].describe()
            stats[f"{col}_by_{target_col}"] = desc

            df.boxplot(column=col, by=target_col, grid=False, figsize=(6, 4))
            plt.title(f"{col} vs {target_col}")
            plt.xlabel(target_col)
            plt.ylabel(col)
            plt.tight_layout()
            if out_dir is not None:
                path = Path(out_dir) / f"{col}_vs_{target_col}.png"
                plt.savefig(path)
                paths[f"{col}_vs_{target_col}"] = str(path)
                plt.close()
            else:
                plt.show()

        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            contingency = pd.crosstab(df[col], df[target_col], normalize="index")
            stats[f"{col}_by_{target_col}"] = contingency.round(2)

            ax = contingency.plot(kind="bar", stacked=True, figsize=(6, 4))
            plt.title(f"{col} vs {target_col}")
            plt.ylabel("Percent")
            plt.xlabel(col)
            plt.legend(title=target_col)
            plt.tight_layout()
            if out_dir is not None:
                path = Path(out_dir) / f"{col}_vs_{target_col}.png"
                ax.get_figure().savefig(path)
                paths[f"{col}_vs_{target_col}"] = str(path)
                plt.close(ax.get_figure())
            else:
                plt.show()

    return stats, paths

def eksplore_regression(df, target_col):
    
    predictors = [col for col in df.columns if col != target_col]

    for col in predictors:
        print(f"\n📊 Variable: {col}")
        
        if pd.api.types.is_numeric_dtype(df[col]):
            # Korelacja liniowa
            corr = df[[col, target_col]].corr().iloc[0, 1]
            print(f"Correlation: {round(corr, 3)}")
            
            # Wykres rozrzutu
            plt.figure(figsize=(5, 4))
            plt.scatter(df[col], df[target_col], alpha=0.6)
            plt.title(f"{col} vs {target_col}\nCorrelation: {round(corr, 3)}")
            plt.xlabel(col)
            plt.ylabel(target_col)
            plt.tight_layout()
            plt.show()

        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            # Średnie wartości targetu w każdej kategorii
            print(df.groupby(col)[target_col].describe())

            # Wykres słupkowy średnich
            df.groupby(col)[target_col].mean().sort_values().plot(kind='bar', figsize=(6, 4))
            plt.title(f"Average {target_col} by {col}")
            plt.ylabel(f"Average {target_col}")
            plt.xlabel(col)
            plt.tight_layout()
            plt.show()
