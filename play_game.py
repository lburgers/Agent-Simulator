#!/usr/bin/env python
import argparse
import glob
import itertools
import numpy as np
from collections import defaultdict

from controller import Controller

SAVE_GIF = True

# these define which parameters are tested (can be changed)
lost_types = ['stationary', 'home', 'route']
tom_types = [True, False]
remembers_types = [True, False]
forget_types = [True, False]
hearing_types = [True, False]

parameters = [lost_types, tom_types, remembers_types, forget_types, hearing_types]
parameter_names = ['lost_types', 'tom_types', 'remembers_types', 'forget_types', 'hearing_types']
sprite_iterator = itertools.product(*parameters)

param_counter = [defaultdict(lambda: 0) for _ in parameters]
sprite_counter = defaultdict(lambda: 0) 

positions = {'A': (8, 3), '0': (22, 13)} 
true_sprite_params = ('home', False, True, True, True)

controller = Controller(positions, true_sprite_params, policy_file="offline_policies.npz")

def load_trial(trial_name):
    gif_filename = glob.glob('./trials/%s/*.gif' % trial_name)[0]
    trial_uuid = gif_filename.split('.gif')[0]

    with open(trial_uuid + '.txt', "r") as f:
        content = f.read()
        import pdb; pdb.set_trace()


def count_match(sprite_params, prob):
    for i, key in enumerate(sprite_params):
        param_counter[i][key] += prob
    sprite_counter[sprite_params] += prob

def marginal_prob(key, dictionary):
    total = sum(dictionary.values())
    if total == 0:
        return 0
    return dictionary[key] / sum(dictionary.values())

def main(trial=None):

    env = controller.make_env(true_sprite_params, [positions['0'], (1, 23)]) #TODO: route assumption here

    action_sequence =  [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0]

    # TODO: trial loader
    # if trial != None:
    #     load_trial(trial)

    state_sequence, action_sequence = controller.run_simulation(action_sequence, human=True, save=False)

    print('TESTING...')
    for sprite_params in sprite_iterator:

        # cant' have TOM without object perm
        if sprite_params[1] and not sprite_params[2]:
            continue

        env = controller.make_env(sprite_params)
        print(sprite_params)
        # if on route sample all corners and test
        
        if sprite_params[0] == 'route':
            home_cords = positions['0']

            corners = controller.sprite.corners

            for corner in corners:

                env = controller.make_env(sprite_params, [home_cords, corner]) # TODO: add more with increasing waypoints
                prob = controller.test_sequence(action_sequence, state_sequence, True)

                if prob > 0:
                    count_match(sprite_params, prob)
                    break

        else:
            prob = controller.test_sequence(action_sequence, state_sequence, True)
            if prob > 0: count_match(sprite_params, prob) 

    print('True sprite params: ', true_sprite_params)
    print('\nPredicted sprite params:')
    for k, v in sprite_counter.items():
        if v > 0: print(k, marginal_prob(k, sprite_counter))


    print('\nMarginal probabilites:\n')
    for i, _ in enumerate(parameters):
        print(parameter_names[i])
        for key in parameters[i]:

            # get route prob from home and stationary
            prob = marginal_prob(key, param_counter[i])
            print(key, ' ', prob),
        print()

    controller.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trial', default=None)
    args = parser.parse_args()

    main(args.trial)
