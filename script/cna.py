import sys
import yaml
from pathlib import Path
from ovito.io import import_file
from ovito.modifiers import CommonNeighborAnalysisModifier


def common_neighbor_analysis(filepath, outpath):

    pipeline = import_file(str(filepath))

    mode = getattr(CommonNeighborAnalysisModifier.Mode, ('IntervalCutoff'))
    pipeline.modifiers.append(CommonNeighborAnalysisModifier(mode=mode))
    data = pipeline.compute(0)

    counter = {}
    for i in data.particles.structure_types:
        counter[i] = counter.get(i, 0) + 1

    with open(outpath, 'w') as f:
        for i in data.particles.structure_types:
            f.write(str(i))
            f.write('\n')
    return None


def pipeline(dirname, time):
    filepath = Path('dumpfiles') / dirname / f'{time}.alloysl'
    cna_path = Path('dataset') / dirname / f'cna/{time}.cna'
    cna_path.parent.mkdir(parents=True, exist_ok=True)
    common_neighbor_analysis(filepath, cna_path)


def main():
    with open(sys.argv[1], 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)
        dirname = CFG['dirname']
        time = CFG['time']

    if type(time) is list:
        times = range(time[0], time[1] + 1, time[2])
        for t in times:
            pipeline(dirname, t)
    else:
        pipeline(dirname, time)


if __name__ == '__main__':
    main()
