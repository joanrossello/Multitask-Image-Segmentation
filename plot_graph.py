"""
Author: Group work
Date: 2021/1/11
Description: plot graphs from csv result
"""

# import packages
import matplotlib.pyplot as plt
import pandas as pd
from utils import plot_figure

# Function: genarate plots from csv
def plot_from_csv(file_path, output_name, model='mtl'):
    df = pd.read_csv(file_path)
    plot_figure(df.to_dict('list'),output_name,model)

if __name__ == '__main__':
    plot_from_csv('results/mtl.csv','mtl.png')
    plot_from_csv('results/mtl_bbox.csv','mtl_bbox.png')
    plot_from_csv('results/mtl_bbox_diff.csv','mtl_bbox_diff.png')
    plot_from_csv('results/mtl_bbox_diff.csv','mtl_bbox_diff.png')
    plot_from_csv('results/mtl_class.csv','mtl_class.png')
    plot_from_csv('results/mtl_class_diff.csv','mtl_class_diff.png')
    plot_from_csv('results/mtl_diff.csv','mtl_diff.png')
    plot_from_csv('results/mtl_diff2.csv','mtl_diff2.png')
    plot_from_csv('results/mtl_recons.csv','mtl_recons.png')
    
    plot_from_csv('results/base0.csv','base_0.png',model='base')
    
    