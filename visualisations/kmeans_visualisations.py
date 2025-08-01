import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.cm as cm

class KMeansVisualizer:
    def __init__(self, output_dir="Results", cluster_dir="ClusterPlots"):
        self.output_dir = pathlib.Path(output_dir)
        self.cluster_dir = pathlib.Path(cluster_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.cluster_dir.mkdir(exist_ok=True)
            
    def plot_with_ci(self,df, x_col, y_col, hue_col, title, xlabel, ylabel, filename):
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, errorbar='ci', marker='o')
    
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
    
        plt.savefig(self.output_dir / filename)
        plt.close()
        print(f"Saved CI plot to {self.output_dir / filename}")
    
    def boxplot_comparison(self, df, x_col, y_col, hue_col, title, xlabel, ylabel, filename):
        output_dir = pathlib.Path(output_dir)
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col)
    
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
        print(f"Saved boxplot to {self.output_dir / filename}")
    
    def plot_hybrid_cap_vs_inertia(self, df):
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
    
        plt.savefig(self.output_dir / "cap_vs_inertia_hybrid.png")
        plt.close()
        print(f"saved: {self.output_dir/ "cap_vs_inertia_hybrid.png"}")
    
    def plot_cap_vs_time(self, df):
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
    
        plt.savefig(self.output_dir / "cap_vs_time_hybrid.png")
        plt.close()
        print(f"Saved: {self.output_dir / "cap_vs_time_hybrid.png"}")
    
    def plot_tolerance_vs_time(self, df):
        output_dir = pathlib.Path(output_dir)
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
        plt.savefig(self.output_dir / "tolerance_vs_time_hybrid.png")
        plt.close()
        print(f"Saved: {self.output_dir / "tolerance_vs_time_hybrid.png"}")
    
    def plot_tolerance_vs_inertia(self, df):
        output_dir = pathlib.Path(output_dir)
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
    
        plt.savefig(self.output_dir / "tolerance_vs_inertia_hybrid.png")
        plt.close()
        print(f"Saved: {self.output_dir / "tolerance_vs_inertia_hybrid.png"}")
    
    def pca_2d_view(self, X_full, centers_full, resolution=300, random_state=0):
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
    
    def plot_clusters(self, X_vis, labels, centers_vis, xx, yy, labels_grid, title="", filename=""):
        plt.figure(figsize=(8, 6))
    
        # Decision boundaries (optional or with low alpha)
        plt.contourf(xx, yy, labels_grid, cmap="Pastel1", alpha=0.2)
    
        # Downsample
        if len(X_vis) > 5000:
            idx = np.random.choice(len(X_vis), size=5000, replace=False)
        else:
            idx = np.arange(len(X_vis))
    
        cmap = cm.get_cmap("tab20", np.unique(labels).size)
    
        plt.scatter(X_vis[idx, 0], X_vis[idx, 1], c=labels[idx], s=8, cmap=cmap, alpha=0.7, edgecolors="none")
        plt.scatter(centers_vis[:, 0], centers_vis[:, 1], c='black', marker='x', s=120, linewidths=2, label="Centers")
    
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.cluster_dir / f"{filename}.png")
        plt.close()
    
