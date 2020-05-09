#!/usr/bin/env python
import os
import argparse
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
from collections import defaultdict

from controller import Controller
from loader import Loader
from vgdl.ontology.constants import *


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



def plot_labels(label):
    states = ('Searching', 'Chasing', 'Intercepting', 'Patrolling', 'Returning', 'Waiting')
    y_pos = np.arange(len(states))

    fig, ax = plt.subplots()

    ax.bar(y_pos, label, align='center', alpha=0.5)
    ax.set_xticklabels(states)
    ax.set_xticks(y_pos)
    ax.set_ylabel('Likelihood')
    ax.set_ylim(0, 1)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

def count_match(sprite_params, prob):
    for i, key in enumerate(sprite_params):

        # if i != 1 or sprite_params[2] == True:
        param_counter[i][key] += prob
            
    sprite_counter[sprite_params] += prob

def marginal_prob(key, dictionary):
    total = sum(dictionary.values())
    if total == 0:
        return 0
    return dictionary[key] / sum(dictionary.values())

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def main(config):

    positions = {'A': (3, 2), '0': (13, 9), 'X': (13, 16)} 
    home_cords = positions['0']
    true_sprite_params = ('home', False, True, True, False)

    direction = LEFT
    if config.dir == 'RIGHT':
        direction = RIGHT

    action_sequence =  [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0]
    state_sequence = None

    # TODO: trial loader
    if config.trial != None:
        loader = Loader(config.trial)
        true_sprite_params = loader.true_params
        action_sequence = loader.player_actions
        positions = loader.locations
        state_sequence = loader.state_sequence

    controller = Controller(positions, true_sprite_params, policy_file=config.policy)

    # if no state sequence generate a new video and state_sequence
    if state_sequence is None or config.save or config.human:
        trial = config.trial
        if trial == None: trial = 'new'
        save_folder_path = './trials/%sv%s' % (trial, config.version)
        os.system('mkdir %s' % save_folder_path)

        env = controller.make_env(true_sprite_params, [positions['0'], (1, 23)], dir=direction, memory_limit=20, hearing_limit=4) #TODO: route assumption here
        state_sequence, action_sequence = controller.run_simulation(action_sequence, state_sequence, human=config.human, save=config.save, save_folder_path=save_folder_path)


    labels = np.zeros((len(state_sequence), 6))

    home_cords = positions['0']

    forgetting_dist = list(range(0, 100, 20))
    forgetting_mean, forgetting_std = 65, 30
    hearing_dist = list(range(1, 9, 2))
    hearing_mean, hearing_std = 5, 2


    print('TESTING...')
    for sprite_params in sprite_iterator:


        print(sprite_params)

        prob = 0
        sprite_labels = np.zeros((len(state_sequence), 6))
        for hearing_limit in hearing_dist:
            for memory_limit in forgetting_dist:

                env = controller.make_env(sprite_params,
                                            [home_cords, (1, home_cords[1]), (1, 23)],
                                            dir=direction,
                                            memory_limit=memory_limit,
                                            hearing_limit=hearing_limit
                                        ) 
                p, sprite_label = controller.test_sequence(action_sequence, state_sequence, False)

                prob += gaussian(hearing_limit, hearing_mean, hearing_std) \
                    * gaussian(memory_limit, forgetting_mean, forgetting_std) \
                    * p
                if sprite_label is not None:
                    sprite_labels += gaussian(hearing_limit, hearing_mean, hearing_std) \
                        * gaussian(memory_limit, forgetting_mean, forgetting_std) \
                        * sprite_label


        if prob > 0:
            count_match(sprite_params, prob)
            labels += (sprite_labels*prob)


    print_string = 'True sprite params: %s\n' % str(true_sprite_params)
    # print_string += '\nPredicted sprite params:\n'

    # match_count = 0
    # for k, v in sprite_counter.items():
    #     if v > 0:

    #         print_string += '%s  %s\n' % ( str(k), str(marginal_prob(k, sprite_counter)) )
    #         match_count += 1


    print_string += '\nMarginal probabilites:\n\n'
    for i, _ in enumerate(parameters):
        print_string += '%s\n' % parameter_names[i]
        for key in parameters[i]:

            # get route prob from home and stationary
            prob = marginal_prob(key, param_counter[i])
            print_string += '%s  $ %.3f $\n' % (key, prob)
        print_string += '\n'

    print(print_string)

    # labels /= float(match_count)
    if save_folder_path:

        with open("%s/%s_posteriors.txt" % (save_folder_path, config.trial), "w") as posterior_file:
            posterior_file.write(print_string)

        imageio.mimsave('%s/%s_labels.gif' % (save_folder_path, config.trial), [plot_labels(l) for l in labels])

        if config.label:
            controller.convert_images_to_mp4(save_folder_path, labels)
        else:
            controller.convert_images_to_mp4(save_folder_path, None)

    else:
        imageio.mimsave('./%s_labels.gif' % config.trial, [plot_labels(l) for l in labels])

    controller.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trial', default=None)
    parser.add_argument('-v', '--version', default=None)
    parser.add_argument('-p', '--policy', default=None)
    parser.add_argument('-d', '--dir', default=None)
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--label', dest='label', action='store_true')
    parser.add_argument('--human', dest='human', action='store_true')
    parser.set_defaults(save=False)
    parser.set_defaults(label=False)
    parser.set_defaults(human=False)
    args = parser.parse_args()

    main(args)
