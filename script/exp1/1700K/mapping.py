import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml


def main():
    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)
        interval = CFG['interval']
        BASE_ENERGY = CFG['BASE_ENERGY']
    csv_path = 'prediction.csv'

    df = pd.read_csv(csv_path)
    truth, pred = df['PotEng'], df['Prediction']
    truth_, pred_ = truth + BASE_ENERGY, pred + BASE_ENERGY
    x = np.linspace(1000, interval * len(df), len(df))

    # plot twin axix data
    fig = plt.figure(figsize=[9, 6])
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(x, truth, label='ground truth', linewidth=1)
    ax1.plot(x, pred, label='prediction', linewidth=1)
    # ax1.set_ylabel('Potential Energy(eV)\nrelative to potential energy at 300fs')

    ax2 = ax1.twinx()
    ax2.plot(x, truth_, label='ground truth', linewidth=1)
    ax2.plot(x, pred_, label='prediction', linewidth=1)
    # ax2.set_ylabel('Potential Energy(eV)\nrelative to infinity')

    plt.title('Experiment1 1700K')
    plt.legend()
    plt.savefig('prediction.png')


if __name__ == '__main__':
    main()
