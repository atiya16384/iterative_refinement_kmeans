import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import pairwise_distances_argmin

def plot_with_ci(df, x_col, y_col, hue_col, title, xlabel, ylabel, filename, output_dir="Results"):
    output_dir = pathlib.Path(output_dir)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, errorbar='ci', marker='o')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path)
    plt.close()
    print(f"Saved CI plot to {output_path}")

def boxplot_comparison(df, x_col, y_col, hue_col, title, xlabel, ylabel, filename, output_dir="Results"):
    output_dir = pathlib.Path(output_dir)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path)
    plt.close()
    print(f"Saved boxplot to {output_path}")

def plot_hybrid_cap_vs_inertia(df, output_dir = "Results"):
    output_dir = pathlib.Path(output_dir)
    df = pd.read_csv(df)
    df_hybrid = df[df["Suite"] == "Hybrid"]
    df_double = df[df["Suite"] == "Double"]
    group_cols = ["DatasetName", "NumClusters", "Cap"]
    df_grouped = df_hybrid.groupby(group_cols)[["Inertia"]].mean().reset_index()


    plt.figure(figsize=(7,5))
    for (ds, k), group in df_grouped.groupby(["DatasetName", "NumClusters"]):
            group_sorted = group.sort_values("Cap")
            base_inertia = df[(df["Suite"] == "Double") & (df["DatasetName"] == ds) & (df["NumClusters"] == k)]["Inertia"].mean()
        
            group_sorted["Inertia"] = group_sorted["Inertia"] / base_inertia
                          
            plt.plot(group_sorted["Cap"], group_sorted["Inertia"], marker = 'o', label=f"{ds}-C{k}")
    
    plt.title("Cap vs Intertia ( Hybrid)")
    plt.xlabel("Cap (Single Precision Iteration Cap)")

    plt.ylabel("Final Inertia")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = output_dir / "cap_vs_inertia_hybrid.png"
    plt.savefig(filename)
    plt.close()
    print(f"saved: {filename}")

def plot_cap_vs_time(df, output_dir="Results"):
    output_dir = pathlib.Path(output_dir)
    df = pd.read_csv(df)
    df_hybrid = df[df["Suite"] == "Hybrid"]
    group_cols = ["DatasetName", "NumClusters", "Cap"]
    df_grouped = df_hybrid.groupby(group_cols)[["Time"]].mean().reset_index()

    plt.figure(figsize=(7,5))
    for (ds, k), group in df_grouped.groupby(["DatasetName", "NumClusters"]):
            
            base_time = df[(df["Suite"] == "Double") & (df["DatasetName"] == ds) & (df["NumClusters"] == k)]["Time"].mean()
            group_sorted = group.sort_values("Cap")
            group_sorted["Time"] = group_sorted["Time"] / base_time
        
            plt.plot(group_sorted["Cap"], group_sorted["Time"], marker = 'o', label=f"{ds}-C{k}")
    
    plt.title("Cap vs Time (Hybrid)")
    plt.xlabel("Cap (Single Precision Iteration Cap)")
    plt.ylabel("Total Time (seconds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = output_dir / "cap_vs_time_hybrid.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_tolerance_vs_time(df, output_dir="Results"):
    output_dir = pathlib.Path(output_dir)
    df = pd.read_csv(df)
    df_hybrid = df[df["Suite"] == "Hybrid"]
    group_cols = ["DatasetName", "NumClusters",  "tolerance_single"]
    df_grouped = df_hybrid.groupby(group_cols)[["Time"]].mean().reset_index()

    plt.figure(figsize=(7, 5))
    for (ds, k), group in df_grouped.groupby(["DatasetName", "NumClusters"]):

        base_time = df[(df["Suite"] == "Double") & (df["DatasetName"] == ds) & (df["NumClusters"] == k) ]["Time"].mean()
        group_sorted = group.sort_values("tolerance_single")
        group_sorted["Time"] = group_sorted["Time"] / base_time

        plt.plot(group_sorted["tolerance_single"], group_sorted["Time"], marker='o', label=f"{ds}-C{k}")

    plt.title("Tolerance vs Time (Hybrid)")
    # plt.ylim(0.99, 1.01)
    plt.xlabel("Single Precision Tolerance")
    plt.xscale('log')
    plt.ylabel("Total Time (seconds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = output_dir / "tolerance_vs_time_hybrid.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_tolerance_vs_inertia(df, output_dir="Results"):
    output_dir = pathlib.Path(output_dir)
    df = pd.read_csv(df)
    df_hybrid = df[df["Suite"] == "Hybrid"]
    group_cols = ["DatasetName", "NumClusters", "tolerance_single"]
    df_grouped = df_hybrid.groupby(group_cols)[["Inertia"]].mean().reset_index()

    plt.figure(figsize=(7, 5))
    for (ds, k ), group in df_grouped.groupby(["DatasetName", "NumClusters"]):
        base_inertia = df[(df["Suite"] == "Double") & (df["DatasetName"] == ds) & (df["NumClusters"] == k) ]["Inertia"].mean()
        
        group_sorted = group.sort_values("tolerance_single")
        group_sorted["Inertia"] = group_sorted["Inertia"] / base_inertia

        plt.plot(group_sorted["tolerance_single"], group_sorted["Inertia"], marker='o', label=f"{ds}-C{k}")

    plt.title("Tolerance vs Inertia (Hybrid)")
    # plt.ylim(0.9999, 1.0001)
    plt.xlabel("Single Precision Tolerance")
    plt.xscale('log')
    plt.ylabel("Final Inertia")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = output_dir / "tolerance_vs_inertia_hybrid.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")


def pca_2d_view(X_full, centers_full, resolution=300, random_state=0):
    pca = PCA(n_components=2, random_state=random_state)
    X_vis = pca.fit_transform(X_full)
    centers_vis = pca.transform(centers_full)

    # Create a meshgrid for decision boundaries
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Assign each grid point to nearest center
  
    labels_grid = pairwise_distances_argmin(grid_points, centers_vis)
    labels_grid = labels_grid.reshape(xx.shape)

    return X_vis, centers_vis, xx, yy, labels_grid

def plot_clusters(X_vis, labels, centers_vis, xx, yy, labels_grid, title="", filename=""):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    # Decision boundary
    plt.contourf(xx, yy, labels_grid, cmap="Pastel1", alpha=0.4)

    # Scatter actual points
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, s=10, cmap="tab10", alpha=0.8, edgecolors="k")
    plt.scatter(centers_vis[:, 0], centers_vis[:, 1], c='black', marker='x', s=100, label="Centers")

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ClusterPlots/{filename}.png")
    plt.close()

