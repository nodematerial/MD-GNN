import os
import sys
import pandas as pd


def file_reader(filepath):
    count = 0
    thermos = ['Step', 'Time', 'Temp', 'PotEng', 'KinEng', 'TotEng', 'Press']
    alldata = []
    with open(filepath)as f:
        while count < 5:
            line = f.readline().rstrip('\n').split()

            if line == thermos:
                data = read_thermo(f)
                alldata.extend(data)

            if line == []:
                count += 1
            else:
                count = 0
    df = pd.DataFrame(alldata, columns=thermos).drop_duplicates()
    return df


def read_thermo(f):
    data = []
    while True:
        line = f.readline().rstrip('\n').split()
        if not line[0].isnumeric():
            break
        data.append(line)
    return data


def main():
    filepath = sys.argv[1]
    savepath = sys.argv[2]
    savedir = os.path.dirname(savepath)
    os.makedirs(savedir, exist_ok=True)

    df = file_reader(filepath)
    df.to_csv(sys.argv[2], index=False)


if __name__ == '__main__':
    main()
