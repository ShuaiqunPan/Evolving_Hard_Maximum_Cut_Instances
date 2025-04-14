import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import os
from networkx import is_connected, from_edgelist
import torch
import umap
from generator import GraphGenerator
import pandas as pd
from utils import is_clique
from config import output_figures_directory, output_others_directory
import random
from maxcut import SDP_max_cut, brute_force_max_cut, random_SDP_max_cut, multiple_random_SDP_max_cuts, run_multiple_rqaoa
from rqaoa import RQAOA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from pandas.plotting import parallel_coordinates
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
from matplotlib.lines import Line2D


def save_figure(filename):
    filepath = os.path.join(output_figures_directory, filename)
    plt.savefig(filepath)

'''
Below is the implementation of ploting the reduction on the calcualted features.
'''
def preprocess_complex_numbers(entry):
    """Helper function to convert complex number strings to floats or handle real values."""
    if isinstance(entry, str):  # Check if the entry is a string
        try:
            # Attempt to convert from string to complex and extract the real part
            return complex(entry.strip('()')).real
        except ValueError:
            return np.nan  # Return NaN for invalid conversions
    elif isinstance(entry, (int, float)):
        return entry  # Directly return the entry if it's already a number
    else:
        return np.nan  # Return NaN for any other unsupported types

# Adjust display settings to show all rows and all columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)

def load_and_preprocess(file_path):
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    initial_shape = df.shape
    print(f"Initial data shape: {initial_shape}")

    complex_cols = ['LOG_LARGESTEIGVAL', 'LOG_SECONDLARGESTEIGVAL', 'LOG_SMALLESTEIGVAL']
    for col in complex_cols:
        df[col] = df[col].apply(preprocess_complex_numbers)
        
    columns_to_drop = ['Unnamed: 0', 'GRAPH_IDX', 'NUM_NODES', 'IS_CONNECTED', 'MIN_ECC', 'MAX_ECC', 'LOGRATIO_MAXMINECC', 'percent_cut', 'percent_positive_lower_triangular',
                       'percent_close1_lower_triangular', 'percent_close3_lower_triangular', 
                       'expected_costGW_over_sdp_cost', 'std_costGW_over_sdp_cost']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    
    final_shape = df.shape
    print(f"Data shape after cleaning: {final_shape}")
    
    print("Remaining columns:", df.columns.tolist())

    if final_shape[0] == 0:
        raise ValueError("All rows have been dropped due to NaNs or inf values.")

    # Ensure the data contains only numeric values before scaling
    numeric_df = df.select_dtypes(include=[np.number])
    problem_rows = numeric_df[numeric_df.isna().any(axis=1) | np.isinf(numeric_df.values).any(axis=1)]
    if not problem_rows.empty:
        print("Rows with NaN or inf values:")
        print(problem_rows)
        raise ValueError("Data still contains infinities or NaNs.")
    
    return df

def combine_data(file_paths):
    labels = adjust_labels(file_paths)
    data_frames = []
    for file_path in file_paths:
        df = load_and_preprocess(file_path)
        df['Label'] = labels[file_paths.index(file_path)]  # Assign label based on file path
        data_frames.append(df)
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined_df.dropna(inplace=True)
    
    filename = os.path.join(output_others_directory, 'combined_data.csv')
    combined_df.to_csv(filename, index=False)
    print("Combined data saved to 'combined_data.csv'")
    return combined_df

def scale_data(combined_df):
    numeric_columns = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns.remove('CALCULATED_RATIO')  # Remove 'CALCULATED_RATIO' from the list of columns to scale
    # scaler = StandardScaler()
    # combined_df[numeric_columns] = scaler.fit_transform(combined_df[numeric_columns])
    # print("Data has been scaled.")
    return combined_df

def log_scale_data(combined_df):
    numeric_columns = combined_df.select_dtypes(include=[np.number]).columns.tolist()

    # List of columns to exclude from log transformation (already log-transformed or special treatment)
    exclude_columns = [
        'CALCULATED_RATIO',
        'LOGRATIO_EDGETONODES',
        'LOG_LARGESTEIGVAL',
        'LOG_SECONDLARGESTEIGVAL',
        'LOG_SMALLESTEIGVAL',
        'LOGRATIO_MAXMINECC',
        'GRAPH_ASSORTATIVITY'
    ]
    
    # Remove excluded columns from the list of columns to log transform
    for col in exclude_columns:
        if col in numeric_columns:
            numeric_columns.remove(col)

    # Apply log transformation to remaining numeric columns, adding a small constant to avoid log(0)
    epsilon = 1e-10  # Small constant to avoid taking log of zero
    for column in numeric_columns:
        combined_df[column] = np.log(combined_df[column] + epsilon)
    print("Data has been log-scaled on selected numeric features.")
    
    filename = os.path.join(output_others_directory, 'combined_log_scale_data.csv')
    combined_df.to_csv(filename, index=False)
    print("Combined data saved to 'combined_data.csv'")
    
    return combined_df

def visualize_umap_global_features(dataframe, n_neighbors=15, min_dist=0.1, n_components=2):
    labels = dataframe['Label'].tolist()
    # Exclude the 'Label' and 'CALCULATED_RATIO' columns for UMAP processing
    data = dataframe.drop(['Label', 'CALCULATED_RATIO'], axis=1).values
    
    # Setup UMAP and fit-transform the data
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    embedding = reducer.fit_transform(data)
    
    fig, ax = plt.subplots()
    unique_labels = list(set(labels))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    label_counts = {label: 0 for label in unique_labels}  # Dictionary to hold total counts per label
    
    scatter_plots = []
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        scatter = ax.scatter(embedding[indices, 0], embedding[indices, 1], label=label, s=10, c=[label_to_color[label]], marker="x")
        scatter_plots.append(scatter)
        label_counts[label] += len(indices)  # Add the count of points per label

    # DBSCAN to find clusters within each group
    dbscan = DBSCAN(eps=0.5, min_samples=10)  # Adjust these parameters as needed
    clusters = dbscan.fit_predict(embedding)
    unique_clusters = np.unique(clusters)
    
    for cluster in unique_clusters:
        if cluster == -1:
            continue  # -1 means noise in DBSCAN
        cluster_points = embedding[clusters == cluster]
        centroid = np.mean(cluster_points, axis=0)
        count = len(cluster_points)
        ax.text(centroid[0], centroid[1], f'n={count}', color='black', fontsize=7)
        
    # Prepare and display the total counts text for each label
    total_counts_text = "\n".join([f'{label}: {count} points' for label, count in label_counts.items()])
    ax.text(0.99, 0.01, total_counts_text.strip(), verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, color='black', fontsize=7)
    ax.legend()
    plt.title('UMAP Projection of Multiple Graph Data Sets')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    save_figure("global_features_umap.pdf")  # Save the figure
    plt.close()
    
def adjust_labels(file_paths):
    adjusted_labels = []
    for path in file_paths:
        # Normalize the path to lower case for case-insensitive comparison
        normalized_path = path.lower()
        
        # Determine the context based on naming convention
        if 'gw_rqaoa' in normalized_path:
            if 'bottom' in normalized_path:
                adjusted_labels.append("Instances with superior RQAOA performance over GW algorithm")
            elif 'top' in normalized_path:
                adjusted_labels.append("Instances with superior GW algorithm over RQAOA performance")
            else:
                # Handle cases where neither 'bottom' nor 'top' is found
                adjusted_labels.append("GW RQAOA Undefined")
        
        elif 'rqaoa_gw' in normalized_path:
            if 'bottom' in normalized_path:
                adjusted_labels.append("Instances with superior GW algorithm over RQAOA performance")
            elif 'top' in normalized_path:
                adjusted_labels.append("Instances with superior RQAOA performance over GW algorithm")
            else:
                # Handle cases where neither 'bottom' nor 'top' is found
                adjusted_labels.append("RQAOA GW Undefined")
        
        else:
            # Default label if none of the specific patterns are matched
            adjusted_labels.append("Unknown Ratio")

    return adjusted_labels

'''
Visualizing multi-dimensional data, showing how feature values vary between different classes.
'''
def plot_parallel_coordinates(dataframe):
    # Ensure the label column is categorical (important for color coding in parallel_coordinates)
    if dataframe['Label'].dtype == 'object':
        dataframe['Label'] = dataframe['Label'].astype('category')
        
    # Drop the 'CALCULATED_RATIO' column if it exists
    if 'CALCULATED_RATIO' in dataframe.columns:
        dataframe = dataframe.drop(columns=['CALCULATED_RATIO'])
        
    # Create a custom colormap for exactly two classes if you know their order
    unique_labels = dataframe['Label'].cat.categories
    color_map = ListedColormap(['#3498db', '#e74c3c'])

    # Create the parallel coordinates plot
    plt.figure(figsize=(20, 8))
    parallel_coordinates(dataframe, class_column='Label', colormap=color_map, alpha=0.5)
    plt.title('Computed Features of Hard Maximum Cut Instances', fontsize=30)
    plt.xlabel('Features', fontsize=22)
    plt.ylabel('Scaled Values', fontsize=22)
    plt.grid(True)
    plt.legend(title='Instance Type', title_fontsize=22, fontsize=18, loc='lower left')
    plt.xticks(rotation=45, fontsize=16)  # Rotate feature names to prevent overlap
    plt.yticks(fontsize=18)
    plt.tight_layout()
    save_figure("parallel_coordinates.pdf")  # Correct way to save the figure
    plt.close()

def plot_parallel_coordinates_two_figures(dataframe):
    # Ensure the label column is categorical (important for color coding in parallel_coordinates)
    if dataframe['Label'].dtype == 'object':
        dataframe['Label'] = dataframe['Label'].astype('category')
        
    # Drop the 'CALCULATED_RATIO' column if it exists
    if 'CALCULATED_RATIO' in dataframe.columns:
        dataframe = dataframe.drop(columns=['CALCULATED_RATIO'])

    # Create a custom colormap for exactly two classes if you know their order
    colors = ['#3498db', '#e74c3c']  # Blue and Red
    unique_labels = dataframe['Label'].cat.categories

    # Create the parallel coordinates plot for each class
    for i, label in enumerate(unique_labels):
        df_subset = dataframe[dataframe['Label'] == label]
        plt.figure(figsize=(20, 8))
        parallel_coordinates(df_subset, class_column='Label', color=colors[i], alpha=0.5)
        plt.title(f'Parallel Coordinates Plot for {label}', fontsize=20)
        plt.xlabel('Features', fontsize=16)
        plt.ylabel('Values', fontsize=16)
        plt.grid(True)
        plt.legend(title='Graph Type', title_fontsize=14, fontsize=14, loc='upper right')
        plt.xticks(rotation=30, fontsize=14)  # Rotate feature names to prevent overlap
        plt.tight_layout()
        save_figure(f"parallel_coordinates_{label}.pdf")  # Save each figure with a label-specific name
        plt.close()


def plot_parallel_coordinates_separate(dataframe):
    if dataframe['Label'].dtype == 'object':
        dataframe['Label'] = dataframe['Label'].astype('category')

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(24, 16), sharex=True)
    categories = ['Instances with superior RQAOA performance over GW algorithm', 
                  'Instances with superior GW algorithm over RQAOA performance']
    titles = ['Graph Instances with Superior RQAOA Performance over GW Algorithm',
              'Graph Instances with Superior GW Algorithm over RQAOA performance']
    colormaps = ['viridis', 'plasma']

    for ax, category, title, colormap in zip(axes, categories, titles, colormaps):
        category_data = dataframe[dataframe['Label'] == category]
        cmap = plt.get_cmap(colormap)

        row_data = pd.Series(dtype=object)
        
        # More precise bounds based on the narrow range
        min_val = category_data['CALCULATED_RATIO'].min()
        max_val = category_data['CALCULATED_RATIO'].max()
        if min_val == max_val:
            bounds = [min_val - 0.01, min_val, min_val + 0.01]
        else:
            bounds = np.linspace(min_val, max_val, 5)
        
        norm = Normalize(vmin=min_val, vmax=max_val)

        for idx, row in category_data.iterrows():
            color = cmap(norm(row['CALCULATED_RATIO']))
            columns_to_drop = ['Label', 'CALCULATED_RATIO']
            if 'Color' in row:
                columns_to_drop.append('Color')
            row_data = row.drop(labels=columns_to_drop)
            ax.plot(row_data.index, row_data.values, color=color, linewidth=1, alpha=0.4)

        ax.set_title(title, fontsize=24)
        ax.set_xlabel('Features', fontsize=16)
        ax.set_ylabel('Scaled Values', fontsize=16)
        ax.grid(True)
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labelsize=12)
        ax.set_xticklabels(row_data.index, rotation=30, fontsize=14)
        ax.set_facecolor('lightgrey')  # Set a background color that contrasts well with your line colors

        # Adjusting colorbar with precise control
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cb.set_label('Calculated Ratio', fontsize=14)
        cb.set_ticks(bounds)
        cb.set_ticklabels([f'{b:.2f}' for b in bounds])  # Increased precision for ticks

    fig.suptitle('Comparative Analysis of Graph Instances (20-node, RQAOA/GW)', fontsize=30, y=0.98)
    fig.subplots_adjust(top=0.92)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure("parallel_coordinates_separate.pdf")
    plt.close()


def plot_parallel_coordinates_separate_two_pdf(dataframe):
    if dataframe['Label'].dtype == 'object':
        dataframe['Label'] = dataframe['Label'].astype('category')

    categories = ['Instances with superior RQAOA performance over GW algorithm', 
                  'Instances with superior GW algorithm over RQAOA performance']
    titles = ['Maximum Cut Instances With Superior RQAOA Performance Over GW Algorithm',
              'Maximum Cut Instances With Superior GW Algorithm Over RQAOA Performance']
    colormaps = ['viridis', 'plasma']
    filenames = ['superior_RQAOA_performance.pdf', 'superior_GW_performance.pdf']

    for category, title, colormap, filename in zip(categories, titles, colormaps, filenames):
        fig, ax = plt.subplots(figsize=(24, 8))  # Adjust figsize if necessary
        category_data = dataframe[dataframe['Label'] == category]
        cmap = plt.get_cmap(colormap)

        row_data = pd.Series(dtype=object)
        
        # More precise bounds based on the narrow range
        min_val = category_data['CALCULATED_RATIO'].min()
        max_val = category_data['CALCULATED_RATIO'].max()
        if min_val == max_val:
            bounds = [min_val - 0.01, min_val, min_val + 0.01]
        else:
            bounds = np.linspace(min_val, max_val, 5)
        
        norm = Normalize(vmin=min_val, vmax=max_val)

        for idx, row in category_data.iterrows():
            color = cmap(norm(row['CALCULATED_RATIO']))
            columns_to_drop = ['Label', 'CALCULATED_RATIO']
            if 'Color' in row:
                columns_to_drop.append('Color')
            row_data = row.drop(labels=columns_to_drop)
            ax.plot(row_data.index, row_data.values, color=color, linewidth=1, alpha=0.4)

        ax.set_title(title, fontsize=30)
        ax.set_xlabel('Features', fontsize=22)
        ax.set_ylabel('Values', fontsize=22)
        ax.grid(True)
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_xticklabels(row_data.index, rotation=45, fontsize=18)

        # Adjusting colorbar with precise control
        colorbar_ax = fig.add_axes([0.93, 0.25, 0.01, 0.7])  # [left, bottom, width, height]
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
        cb.set_label('Calculated Ratio', fontsize=22)
        cb.set_ticks(bounds)
        cb.set_ticklabels([f'{b:.3f}' for b in bounds], fontsize=18)  # Increased precision for ticks

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        save_figure(filename)  # Save each figure to a separate file
        plt.close(fig)  # Close the figure after saving to free up memory