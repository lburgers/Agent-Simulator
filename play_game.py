#!/usr/bin/env python

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

positions = {'A': (3, 3), '0': (22, 13)}
true_sprite_params = ('home', False, True, False, False)


controller = Controller(positions, true_sprite_params)
search_controller = Controller(positions, true_sprite_params, pref="search")

def count_match(sprite_params):
    for i, key in enumerate(sprite_params):
        param_counter[i][key] += 1
    sprite_counter[sprite_params] += 1

def marginal_prob(key, dictionary):
    total = sum(dictionary.values())
    if total == 0:
        return 0
    return dictionary[key] / sum(dictionary.values())

def check_match_with_search(test_controller, actions, true_sequence, sprite_params):
    test_sequence, _ = test_controller.run_simulation(actions)

    if true_sequence.shape == test_sequence.shape and (true_sequence == test_sequence).all():
        return True
    
    searching = False
    match = True

    jdiff = 0
    search_index = 2
    searching_counter = 0
    max_search_len = 0

    sprite = test_controller.sprite


    # test if searching
    for j in range(len(true_sequence)):

        # get correct states TODO: check that this works for forgetting
        if not searching and j - jdiff >= len(test_sequence):
            match = False
            break
        elif searching and j - jdiff >= len(test_sequence):
            true_state, test_state = true_sequence[j], np.zeros(len(true_sequence[j]))
        else:
            true_state, test_state = true_sequence[j], test_sequence[j - jdiff]

        # run tests
        if (true_state == test_state).all():
            continue

        elif (searching or true_state[search_index] == test_state[search_index]) and true_state[search_index] == 1:
            searching = True
            searching_counter += 1
            continue

        elif searching and true_state[search_index] == 0:
            max_search_len = max(max_search_len, searching_counter)
            searching_counter = 0
            searching = False
            search_controller.build_map({
                'A': (int(true_sequence[j-1][3]), int(true_sequence[j-1][4])),
                '0': (int(true_sequence[j-1][0]), int(true_sequence[j-1][1])),
                })
            search_env = search_controller.make_env(sprite_params, home=positions['0'])
            search_state_sequence, _ = search_controller.run_simulation(actions[j:])
            jdiff = j
            test_sequence = search_state_sequence
            search_controller.close()

            if not (true_sequence[j] == test_sequence[j - jdiff]).all():
                match = False
                break
        else:
            match = False
            break

    if sprite.forgets and max_search_len >= sprite.memory_limit:
        match = False

    return match


def main():

    env = controller.make_env(true_sprite_params, [positions['0'], (1, 23)]) #TODO: route assumption here

    action_sequence = [4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 4, 0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0, 0]
    state_sequence, action_sequence = controller.run_simulation([], human=True, save=True)

    print('TESTING...')
    for sprite_params in sprite_iterator:

        env = controller.make_env(sprite_params)

        # if on route sample all corners and test
        if sprite_params[0] == 'route':
            home_cords = positions['0']

            corners = controller.sprite.corners
            for corner in corners:

                env = controller.make_env(sprite_params, [home_cords, corner]) # TODO: add more with increasing waypoints

                if check_match_with_search(controller, action_sequence, state_sequence, sprite_params):
                    print('match', sprite_params)
                    count_match(sprite_params)
                    break

        elif check_match_with_search(controller, action_sequence, state_sequence, sprite_params):
            print('match', sprite_params)
            count_match(sprite_params)            

    print('True sprite params: ', true_sprite_params)
    print('\nPredicted sprite params:')
    for k, v in sprite_counter.items():
        if v > 0: print(k)


    print('\nMarginal probabilites:\n')
    for i, _ in enumerate(parameters):
        print(parameter_names[i])
        for key in parameters[i]:

            # get route prob from home and stationary
            prob = marginal_prob(key, param_counter[i])
            print(key, ' ', prob),
        print()

    controller.close()
    search_controller.close()

if __name__ == '__main__':
    main()
