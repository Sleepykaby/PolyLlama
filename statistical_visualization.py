#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ast
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from typing import (
    Tuple,
    Dict
)


def update_plot_styles(style_dict: Dict = None):
    """ Update global drawing styles for the plots based on a provided dictionary"""
    default_style = {
        'font.family': 'Arial',
        'axes.titlesize': 30,
        'axes.labelsize': 30,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 18,
        'axes.linewidth': 3,
        'legend.frameon': False,
        'legend.loc': 'upper right',
        'legend.labelspacing': 0.3,
        'legend.handletextpad': 0.5,
        'axes.labelpad': 5
    }
    
    if style_dict is not None:
        # Update default styles with any provided overrides
        default_style.update(style_dict)
    
    plt.rcParams.update(default_style)


class DatasetViewer:
    """visualization of feature distribution of dataset"""
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        _headers = set(self.df.columns)
        _required_columns = {"context length", "token type", "total count", "one hot count"}
        assert _required_columns.issubset(_headers), f"Missing column: {_required_columns.difference(_headers)}"
        
    def statistic_bar(self, title) -> None:
        """Generate a bar chart of the one hot count statistics."""
        COUNT = 0
        total_counter = Counter()

        for count_str in self.df['one hot count']:
            count_dict = ast.literal_eval(count_str)  # Transform string into dict
            total_counter.update(count_dict)
            COUNT += 1

        labels = list(total_counter.keys())
        total = list(total_counter.values())
        percentages = [count / COUNT * 100 for count in total]

        # Create bar chart
        _, ax = plt.subplots(figsize=(7.5, 6))
        bars = ax.bar(labels, percentages, color="royalblue")

        # Add percentage labels on bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.1f}%', va='bottom', ha='center', fontsize=18)

        # Set chart limits and labels
        ax.set_ylim(0, 100)
        if title is not None:
            ax.set_title(f'{title}')
        ax.set_xlabel('Entities')
        ax.set_ylabel('Percentage (%)')
        ax.tick_params(axis='x', which='major', length=15, width=3, labelsize=18)
        ax.tick_params(axis='y', which='major', length=15, width=3)

        plt.savefig(f'bar.tif', dpi=300, bbox_inches='tight')
        plt.close()

    def one_hot(self):
        df = self.df.loc[:, 'T_melt':'t_hold']
        num_rows = df.shape[0]    # Count rows
        num_cols = df.shape[1]    # Count columns
        min_val = df.values.min()
        max_val = df.values.max()
        # Display
        plt.figure(figsize=(14, 5))
        # Set mapping of color
        n_of_cmap = max_val - min_val + 1
        cmap = plt.get_cmap('Blues', n_of_cmap)  # One of 'YlGnBu' and 'Blues' and 'Reds'
        # Normalization of cmap
        boundaries = range(min_val, n_of_cmap + 1)
        norm = BoundaryNorm(boundaries, cmap.N)
        # Display
        heatmap = plt.imshow(df.T, cmap=cmap, norm=norm, aspect='auto')  # Role of 'interpolation'
        # Set the format of x and y labels
        plt.yticks(range(num_cols), df.columns)
        plt.xticks([0, num_rows], labels=['0', str(num_rows)])
        plt.tick_params(axis='both', which='major', direction="out", length=15, width=3)

        # Display colorbar
        cbar = plt.colorbar(
            heatmap, ticks=range(min_val + 1, max_val + 2, 1), label='Value', 
            drawedges=True, aspect=15, shrink=1, fraction=0.15, pad=0.02
        )
        cbar.outline.set_linewidth(3)
        cbar.ax.tick_params(labelsize=24, length=10, width=3)
        cbar.ax.set_ylabel("Count", fontsize=30)
        
        # Automatically adjust layout
        plt.savefig(f'heatmap.tif', dpi=300, bbox_inches='tight')
        plt.close()

    def statistic_scatter(self):
        # Adjust as your need
        kwargs = {
            "ax1": (self.df["context length"], self.df["token type"], "Context Length", "Token Type", 
                        (300, 1100), (100, 300), (300, 1100), (100, 300), 200, 50),
            "ax2": (self.df["context length"], self.df["total count"], "Context Length", "Count", 
                    (300, 1100), (-0.5, 6.5), (300, 1100), (0, 6), 200, 1),
            "ax3": (self.df["token type"], self.df["total count"], "Token Type", "Count", 
                    (100, 300), (-0.5, 6.5), (100, 300), (0, 6), 50, 1)
        }
        
        # Setting axes
        _, axs = plt.subplots(nrows=1, ncols=3, figsize=(22.5, 6.5))
        
        # Draw
        for ax, (_, (x, y, x_label, y_label, x_lim, y_lim, x_locator, y_locator, x_interval, y_interval)) in zip(axs, kwargs.items()):
            self.__plot_scatter(ax, x, y, x_label, y_label, x_lim, y_lim, x_locator, y_locator, x_interval, y_interval)

        # Compact layout and presentation
        plt.tight_layout()
        plt.savefig(f'scatter.tif', dpi=300, bbox_inches='tight')
        plt.close()

    def __plot_scatter(self, ax, x, y, x_label: str, y_label: str, x_lim: Tuple[int, int], y_lim: Tuple[int, int], x_locator: Tuple[int, int], y_locator: Tuple[int, int], x_interval: int, y_interval: int):
        ax.scatter(x, y, edgecolor="black", facecolor="dodgerblue", alpha=1, s=200, linewidths=1.5, clip_on=False)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
        ax.xaxis.set_major_locator(plt.FixedLocator(np.arange(x_locator[0], x_locator[1]+1, x_interval)))
        ax.yaxis.set_major_locator(plt.FixedLocator(np.arange(y_locator[0], y_locator[1]+1, y_interval)))
        ax.tick_params(axis='both', direction="out", length=15, width=3)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
