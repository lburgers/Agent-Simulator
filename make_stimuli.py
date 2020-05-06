import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--version', default=None)
args = parser.parse_args()

file_names = [str(x) for x in range(0, 15)]

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


# TODO: change this on next use (it has unique tweaks in it)
for name in file_names:
    # name = '%sv5' % name
    direction = 'LEFT'
    if name[0] == '3':
        direction = 'RIGHT'
    
    # os.system('python play_game.py -t %s -v %s -p offline_policies.npz --save --dir %s' % (name, args.version, direction))
    # os.system('python play_game.py -t %s -v %s -p relaxed_policies.npz --save --dir %s' % (name, args.version, direction))
    with open('./trials/%s/%sv%s_posteriors.txt' % (name, name, args.version), 'r') as f:
        content = f.read()
        splits = content.split('$')
        trial_data = [float(splits[i].strip()) for i in posterior_keys.values()]
        full_data.append(trial_data)


df = pd.DataFrame(full_data, columns = list(posterior_keys.keys()))
df.to_csv('./trials/marginal_posterior_v8.csv')
