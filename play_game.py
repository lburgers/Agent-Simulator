#!/usr/bin/env python

import itertools
from collections import defaultdict

from controller import Controller

SAVE_GIF = True

# these define which parameters are tested (can be changed)
lost_types = ['stationary', 'home', 'static_route', 'search']
tom_types = [True, False]
remembers_types = [True, False]
hearing_types = [True, False]

parameters = [lost_types, tom_types, remembers_types, hearing_types]
sprite_iterator = itertools.product(*parameters)


def main():

    # positions = {'A': (20,3), '0': (20,13), 'Z': (20,21), 'X': (1,21)}
    positions = {}
    true_sprite_params = ('static', False, True, False)    

    controller = Controller(positions, true_sprite_params)
    env = controller.make_env(true_sprite_params)

    action_sequence = [0] * 6
    action_sequence += [2] * 11
    action_sequence += [1] * 16
    action_sequence += [3] * 12
    state_sequence = controller.run_simulation(action_sequence, human=True, save=False)

    import pdb; pdb.set_trace()

    param_counter = [defaultdict(lambda: 0) for _ in parameters]
    sprite_counter = defaultdict(lambda: 0)

    print('TESTING...')
    for sprite_params in sprite_iterator:

        env = controller.make_env(sprite_params)

        test_state_sequence = controller.run_simulation(action_sequence)

        if state_sequence.shape == test_state_sequence.shape and (state_sequence == test_state_sequence).all():
            for i, key in enumerate(sprite_params):
                param_counter[i][key] += 1
            sprite_counter[sprite_params] += 1

    print('True sprite params: ', true_sprite_params)
    print('\nPredicted sprite params:')
    for k, v in sprite_counter.items():
        if v > 0: print(k)

    print('\nMarginal probabilites:')
    for i, _ in enumerate(parameters):
        for key in parameters[i]:
            marginal_prob = param_counter[i][key] / sum(param_counter[i].values())
            print(key, ' ', marginal_prob),
        print()

    controller.close()

if __name__ == '__main__':
    main()
