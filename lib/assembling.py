import pandas as pd
from utils import cleaning_data, cond_plot
from model import PredictionGeneration
from itertools import product
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description='Parser for training model')

parser.add_argument('--windows', nargs='+', type=int, required=True, help='List of integers for windows')
parser.add_argument('--out_vals', nargs='+', type=int, required=True, help='List of integers for out_vals')
parser.add_argument('--conv', action='store_true', help='Boolean flag for conv')
parser.add_argument('--shuffle', action='store_true', help='Boolean flag for shuffle')

args = parser.parse_args()

windows = args.windows
out_vals = args.out_vals
conv = args.conv
shuffle = args.shuffle
print(conv)
print(shuffle)

conv_list = [True, False] if conv else [True]
shuffle_list = [True, False] if shuffle else [True]

df_data = pd.read_csv('../data/train.csv')
df_submission = pd.read_csv('../data/sample_submission.csv')

df_data = cleaning_data(df_data)
vals_product = product(windows,out_vals,conv_list,shuffle_list)
total = len(list(vals_product))
vals_product = product(windows,out_vals,conv_list,shuffle_list)
dataframe_list = []
for i, (window, out_vals, conv_val, shuffle_val) in enumerate(vals_product):
    name = f'{i+1}/{total} win_{window} out_vals_{out_vals} conv_{conv_val} shuffle_{shuffle_val}'
    dataframe_list.append(cond_plot(PredictionGeneration,df_data, df_submission, window, out_vals, conv_val, shuffle_val, name))
df_res = pd.concat(dataframe_list)
df_res.to_csv('../results/results.csv',index=False)
df_res['index'] = list(range(419)) * total

if shuffle and conv:
    g = sns.FacetGrid(data = df_res.query('shuffle == True'), col='window',row='out_vals')
    g.map_dataframe(sns.pointplot,y='sleep_hours',x='index',hue='conv')
    g.add_legend()
    g.savefig('../results/shuffle_true.png')

    g = sns.FacetGrid(data = df_res.query('shuffle == False'), col='window',row='out_vals')
    g.map_dataframe(sns.pointplot,y='sleep_hours',x='index',hue='conv')
    g.add_legend()
    g.savefig('../results/shuffle_false.png')
else:
    g = sns.FacetGrid(data=df_res, col='window', row='out_vals')
    if shuffle + conv == 1:
        param = 'shuffle' * int(shuffle) + 'conv' * int(conv)
        g.map_dataframe(sns.pointplot, y='sleep_hours', x='index', hue=param)
        g.add_legend()
    else:
        g.map_dataframe(sns.pointplot, y='sleep_hours', x='index')
    g.savefig('../results/results.png')

min_loss = df_res['val_loss'].min()
df_res.query("val_loss == @min_loss").drop(['shuffle','conv','window','out_vals','index','val_loss'],axis=1)\
    .to_csv('../results/submission.csv',index=False)
min_window = df_res.query("val_loss == @min_loss")['window'].unique()[0]
min_outval = df_res.query("val_loss == @min_loss")['out_vals'].unique()[0]
min_shuffle = df_res.query("val_loss == @min_loss")['shuffle'].unique()[0]
min_conv = df_res.query("val_loss == @min_loss")['conv'].unique()[0]
print(f'Sumbission was made with val_loss = {min_loss} window {min_window} out_val {min_outval} conv {min_conv} shuffle {min_shuffle}')