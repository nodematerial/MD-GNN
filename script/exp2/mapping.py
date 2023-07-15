import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os


def make_plot(x, title, truth, pred):
    fig = plt.figure(figsize=[9, 6])
    ax1 = fig.add_subplot(1, 1, 1)

    # plor settings
    ax1.plot(x, truth, label='ground truth', linewidth=1.6)
    ax1.plot(x, pred, label='prediction', linewidth=1.6)

    # axis settings
    ax1.tick_params(axis='x', direction='in', labelsize=16)
    ax1.tick_params(axis='y', direction='in', labelsize=16)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(30000))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(250))

    # other component settings
    plt.title(title)
    plt.legend()
    return plt


def main():
    os.makedirs('images', exist_ok=True)

    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)
        interval = CFG['interval']
        exp_name = CFG['exp_name']
        all_dirs = CFG['all_dirs']
        data_per_conf = CFG['data_per_condition']
        BASE_ENERGY = CFG['BASE_ENERGY']

    csv_path = 'prediction.csv'
    df = pd.read_csv(csv_path)

    for i, name in enumerate(all_dirs):
        x = np.linspace(1000, interval * data_per_conf, data_per_conf)
        title = f'{exp_name}_{name}'
        truth = df['PotEng'].iloc[data_per_conf * i: data_per_conf * (i + 1)] + BASE_ENERGY[i]
        pred = df['Prediction'].iloc[data_per_conf * i: data_per_conf * (i + 1)] + BASE_ENERGY[i]

        plt = make_plot(x, title, truth, pred)
        plt.savefig(f'images/{exp_name}_{i}.png')


if __name__ == '__main__':
    main()
