#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
from typing import (
    List,
    Dict,
    Any
)


def semantic_embedding(tokenizer: Any, base_model: Any, group_sents: List[str], model_name:str):
    # Tokenize the 'context' column of datasets
    encoded_input = tokenizer(
        text = group_sents,
        padding = True,  # padding each one to the length of the longest sample in set 
        return_tensors = "pt"
    ).to('cuda:0')
    
    # Embedding tokens
    with torch.no_grad():
        # **1. LLaMA version
        if model_name == 'llama2':
            input_embeds = base_model.model.embed_tokens(encoded_input['input_ids']).to('cuda:0')
        # **2. BERT version
        elif model_name == 'bert':
            input_embeds = base_model(**encoded_input)[0].to('cuda:0')

    # Expand mask matrix to the same size of embeds
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(input_embeds.size()).float().to(input_embeds.device)

    # The embeds of each token is weighted and averaged according to each dimension to obtain the embeds of the sentence
    sentence_embeds = torch.sum(input_embeds * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9).to('cuda:0')
    sentence_normalized_embeds = torch.nn.functional.normalize(sentence_embeds, p=2, dim=1)
    
    # Transform the tensor into array
    embeds_array = sentence_normalized_embeds.cpu().numpy()
    
    return embeds_array


def color_coding(embeddings: Any, one_hot_dicts: Dict[str, int]):
    # T-SNE to reduce dimensionality for visualization
    tsne = TSNE(n_components=2, perplexity=5, random_state=66)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create a figure and axis for plotting
    _, ax = plt.subplots(figsize=(7.5, 6))

    ax.set_facecolor('#FFFFF7')

    # Define colors for each parameter
    colors = {
        'melt temperature': 'red',
        'mold temperature': 'blue',
        'injection speed': 'green',
        'injection pressure': 'purple',
        'holding pressure': 'orange',
        'holding time': 'cyan'
    }

    # Loop through each embedding to plot
    for i in range(len(embeddings_2d)):
        one_hot = one_hot_dicts[i]                  # Get the one-hot encoded dictionary for the current point
        total_non_zero = sum(one_hot.values())      # Calculate the total count of non-zero features
        
        if total_non_zero == 0:
            # If no parameters are present, plot as a gray point
            ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], s=50, c='#CFCFCF')
        else:
            # Calculate the ratios for the pie chart
            start_angle = 0
            ratios = [one_hot[param] / total_non_zero for param in one_hot.keys()]
            
            for ratio, color in zip(ratios, colors.values()):
                if ratio > 0:
                    end_angle = start_angle + ratio * 360  # Calculate the end angle for the wedge
                    wedge = Wedge((embeddings_2d[i, 0], embeddings_2d[i, 1]), r=1.5, 
                                  theta1=start_angle, theta2=end_angle, color=color, alpha=0.8)
                    ax.add_patch(wedge)         # Add the wedge to the plot
                    start_angle = end_angle     # Update the start angle for the next wedge

    # Create legend elements for the plot
    legend_elements = []

    for label, color in colors.items():
        patch = Patch(facecolor=color, label=label) 
        legend_elements.append(patch)

    null_patch = Patch(facecolor='#CFCFCF', label='Null')
    legend_elements.append(null_patch)  # Add a 'Null' category for missing parameters
    ax.legend(handles=legend_elements, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3)

    # Set limits for x and y axes
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the plot
    plt.savefig(f'color_coding.tif', dpi=300, bbox_inches='tight')
    plt.close()


def hierarchical_cluster(embeddings: Any, perplexity: float = 5, distance_threshold: float = 50):
    """ Perform T-SNE and Agglomerative Clustering on embeddings.
    params:
    'perplexity': T-SNE perplexity, higher values might help capture global structure.
    'distance_threshold': The linkage distance threshold for merging clusters.
    """
    # T-SNE to reduce dimensionality for visualization
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=66)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Using cosine distances for Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage='ward')
    cluster_labels = clustering.fit_predict(embeddings_2d)
    n_clusters = len(set(cluster_labels))

    return embeddings_2d, cluster_labels, n_clusters


def tsne_visualization(embeddings_2d: Any, one_hot_dicts: Any, cluster_labels: Any, n_clusters: int, param: str, subset_label: List[int]):
    """ Plot T-SNE visualization for each process parameter. """
    # cluster_colors = plt.get_cmap('tab20c', n_clusters)
    cluster_colors = [
        '#2E75B6', '#BDD7EE', '#C55A11', '#F8CBAD', '#D45EB2', '#A5C7C4', 
        '#C09316', '#F0D27C', '#7030A0', '#CBA9E5', '#767171', '#BFBFBF', 
        '#A9D18E', '#C00000', '#81D2DF', '#C17D7D', '#518580', '#EEC0E1'
    ]
    
    contains_param_idx = []
    not_contains_param_idx = []

    for i, one_hot in enumerate(one_hot_dicts):
        if one_hot[param] != 0 and i in subset_label:
            contains_param_idx.append(i)
        else:
            not_contains_param_idx.append(i)
    

    plt.figure(figsize=(7.5, 6))
    ax = plt.gca()

    # Non-parameter points
    ax.scatter(embeddings_2d[not_contains_param_idx, 0], embeddings_2d[not_contains_param_idx, 1], 
                facecolors='none', edgecolors='gray', s=120, linewidths=1)

    # Parameter points
    for cluster in range(n_clusters):
        cluster_points_idx = [i for i in contains_param_idx if cluster_labels[i] == cluster]
        ax.scatter(embeddings_2d[cluster_points_idx, 0], embeddings_2d[cluster_points_idx, 1], 
                    facecolors=cluster_colors[cluster], edgecolor='black', s=120, linewidths=1, alpha=1.0)

    ax.set_title(f'T-SNE plot for {param}', pad=10)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.tick_params(axis='both', direction="out", length=15, width=1.5, pad=6)
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.xaxis.set_major_locator(plt.FixedLocator(range(-80, 81, 40)))
    ax.yaxis.set_major_locator(plt.FixedLocator(range(-80, 81, 40)))

    plt.savefig(f'{param}.tif', dpi=300, bbox_inches='tight')
    plt.close()


def calculate_silhouette_scores(X, cluster_labels):
    """ Calculate silhouette scores for clustering."""
    silhouette_values = silhouette_samples(X, cluster_labels)
    silhouette_avg = silhouette_score(X, cluster_labels)
    return silhouette_values, silhouette_avg


def display_silhouette(silhouette_values, silhouette_avg, cluster_labels):
    """Plot silhouette scores for each sample, organized by cluster without cluster labels."""
    cluster_colors = [
        '#2E75B6', '#BDD7EE', '#C55A11', '#F8CBAD', '#D45EB2', '#A5C7C4', 
        '#C09316', '#F0D27C', '#7030A0', '#CBA9E5', '#767171', '#BFBFBF', 
        '#A9D18E', '#C00000', '#81D2DF', '#C17D7D', '#518580', '#EEC0E1'
    ]

    n_clusters = len(np.unique(cluster_labels))
    _, ax = plt.subplots(figsize=(7, 5))
    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cluster_colors[i % len(cluster_colors)]
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        y_lower = y_upper  # no gap between clusters
    
    ax.set_title(f"Average Silhouette Values: {silhouette_avg: .2f}")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=np.mean(silhouette_values), color="red", linestyle="--")
    ax.set_yticks([])  # Clear the y-axis labels / ticks
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.tick_params(axis='both', direction="out", length=15, width=1.5, pad=6)

    plt.show()
