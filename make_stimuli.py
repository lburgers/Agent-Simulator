import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--version', default=None)
args = parser.parse_args()

file_names = '0 1_diff 2 3 4 5_slight_diff 6 7 8 9_diff 10_diff 11 12 13_todo 14'
file_names = file_names.split(' ')

posterior_keys = {
    'stationary': 1,
    'home': 3,
    'route': 5,
    'tom': 7,
    'memory': 11,
    'forgets': 15,
    'hears': 19,
}

full_data = []

for name in file_names:
    
    os.system('python play_game.py -t %s -v %s -p offline_policies.npz --save' % (name, args.version))
    with open('./trials/%sv%s/%s_posteriors.txt' % (name, args.version, name), 'r') as f:
        content = f.read()
        splits = content.split('$')
        trial_data = [float(splits[i].strip()) for i in posterior_keys.values()]
        full_data.append(trial_data)


df = pd.DataFrame(full_data, columns = list(posterior_keys.keys()))
df.to_csv('./trials/marginal_posterior_v%s.csv' % args.version)
