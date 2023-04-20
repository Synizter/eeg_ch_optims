#find start and stop of block data
from itertools import groupby
from operator import itemgetter


fs = 500
durations = 2
sample = fs * durations
ch = 19

def butterworth_bandpass(low, high, fs, order = 4):
    return butter(order, [low, high], fs = fs, btype = 'band', output = 'ba', analog = False)

def butterworth_bpf(data, low, high, fs, order = 4):
    b,a = butterworth_bandpass(low, high, fs / 2, order)
    return lfilter(b,a, data, axis = -1)

def find_target_block_loc(e):
    evt = np.where(e == 1)[0]
    ranges =[]

    for k,g in groupby(enumerate(evt),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        ranges.append((group[0],group[-1]))
    return ranges

def extract_target_class(seq, target, ohe_num, total_class = 7):
    #get target block in list of tuple
    locs = find_target_block_loc(target)
    tmp = []
    for loc in locs:
        block_data = np.expand_dims(seq[loc[0]:loc[1]], axis = 0)
        tmp.append(block_data)
    data = np.vstack([t for t in tmp])
    #create ohe of target
    label = np.zeros((data.shape[0], total_class))
    label[:, ohe_num - 1] += 1
    print(label)
    return data, label 