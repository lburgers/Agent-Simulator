#!/usr/bin/env python

import os
import shutil
import logging
import time
import itertools
from collections import defaultdict
import numpy as np

import imageio
import png

import gym
from gym.envs.registration import register
import vgdl.interfaces.gym
from vgdl.util.humanplay.controls import VGDLControls

from build_level import BuildLevel
from additional_sprites import CustomAStarChaser

SAVE_GIF = True


def register_vgdl_env(domain_file, level_file, observer=None, blocksize=None, counter=0):
    level_name = '.'.join(os.path.basename(level_file).split('.')[:-1])
    env_name = 'vgdl_{}-v0'.format(counter)

    register(
        id=env_name,
        entry_point='vgdl.interfaces.gym:VGDLEnv',
        kwargs={
            'game_file': domain_file,
            'level_file': level_file,
            'block_size': blocksize,
            'obs_type': observer,
        },
    )

    return env_name

# these define which parameters are tested (can be changed)
agent_types = ['search', 'random', 'stationary']
search_goals = ['avatar']
search_types = ['approach', 'avoid']
lost_types = ['random', 'stationary']
sight_radii = [5, 10, 25]
oriented_views = [True, False]
remembers_types = [True, False]
walls_types = [True, False]

parameters = [agent_types, search_goals, search_types, lost_types, sight_radii,
                oriented_views, remembers_types, walls_types]
sprite_iterator = itertools.product(*parameters)

# build sprite class with custom properties
def make_sprite(agent_type='search', search_goal='avatar', search_type='avoid',
                lost_type='random', sight_radius=10, oriented_view=True, remembers=True, walls=True):

    class CustomNPC(CustomAStarChaser):
        speed = 1

        # base npc planner
        search = True if agent_type == 'search' else False
        stationary = True if agent_type == 'stationary' else False
        random = True if agent_type == 'random' else False

        # if search define target, avoid/approach, and what to do when lost
        target = search_goal # can be [avatar, A, B, C]
        fleeing = True if search_type == 'avoid' else False
        lost_function = lost_type # can be [random, stationary]

        # if search define perception params
        see_through_walls = walls
        sight_limit = sight_radius
        full_field_view = not oriented_view
        memory = remembers

    return CustomNPC

def build_map(levelfile):
    builder = BuildLevel(levelfile)
    builder.add(20,5, 'A') # Add an avatar to the map at (x, y)
    builder.add(18,2, '0') # Add a friendly (reward +1) NPC to the map
    # builder.add(11,9, '1') # Add a dangerous (reward -1) NPC to the map
    builder.add(20,21, 'Z') # Add a GOAL to the map
    # builder.add(1,9, 'Y') # Add a GOAL to the map
    builder.add(1,21, 'X') # Add a GOAL to the map
    builder.save()
    return builder

def run_simulation(action_sequence, env, trial_count=None):
    states = np.array([])
    env.reset()
    step = 0
    for step_i in itertools.count():
        # controls.capture_key_presses()
        # env.render()
        obs, reward, done, info = env.step(action_sequence[step_i])

        if trial_count != None:
            rgb_array = env.render('rgb_array')
            rgb_shape = rgb_array.shape
            png.from_array(rgb_array.reshape(-1, 3*rgb_shape[1]), 'RGB').save("./trials/%d/%d.png"% (trial_count, step))
            step += 1

        if done or step_i == len(action_sequence) - 1:
            break

        states = [obs] if len(states) == 0 else np.vstack((states, obs))

        # time.sleep( 1/ 15.0)
    return states

def main():

    levelfile = './level.txt'
    domainfile = './game.txt'
    observer_cls = 'objects'
    blocksize = 24
    reps = 1
    pause_on_finish = False
    tracedir = None
    env_counter = 0

    trial_count = 10

    builder = build_map(levelfile)

    # TODO: add code for making multiple sprites and adding them to game.txt
    # TODO: add rewards (friendly/unfriendly)
    true_sprite_params = ('search', 'avatar', 'approach', 'stationary', 25, False, True, False)
    sprite = make_sprite(
            agent_type = true_sprite_params[0],
            search_goal = true_sprite_params[1],
            search_type = true_sprite_params[2],
            lost_type = true_sprite_params[3],
            sight_radius = true_sprite_params[4],
            oriented_view = true_sprite_params[5],
            remembers = true_sprite_params[6],
            walls = true_sprite_params[7],
        )

    vgdl.registry.register(sprite.__name__, sprite)
    env_name = register_vgdl_env(domainfile, builder.level_name, observer_cls, blocksize, env_counter)
    env_counter += 1
    # logging.basicConfig(format='%(levelname)s:%(name)s %(message)s',
    #         level=logging.DEBUG)
    # logger = logging.getLogger(__name__)

    env = gym.make(env_name)
    fps = 15
    cum_reward = 0
        
    # controls = VGDLControls(env.unwrapped.get_action_meanings())
    # env.render('human')
    if SAVE_GIF:
        basedir = './trials/%d/' % trial_count
        if os.path.exists(basedir):
            shutil.rmtree(basedir) 
        os.mkdir(basedir)

    action_sequence = [2] * 11
    action_sequence += [1] * 14
    action_sequence += [3] * 12
    state_sequence = run_simulation(action_sequence, env, trial_count=trial_count)

    if SAVE_GIF:
        images = []
        for i in range(len(os.listdir(basedir))-1):
            images.append(imageio.imread(basedir + '%d.png' % (i)))
        imageio.mimsave('./trials/%d.gif' % trial_count, images)
        shutil.rmtree(basedir)

    param_counter = [defaultdict(lambda: 0) for _ in parameters]
    sprite_counter = defaultdict(lambda: 0)

    print('TESTING...')
    import pdb; pdb.set_trace()
    for sprite_params in sprite_iterator:

        sprite = make_sprite(
                agent_type = sprite_params[0],
                search_goal = sprite_params[1],
                search_type = sprite_params[2],
                lost_type = sprite_params[3],
                sight_radius = sprite_params[4],
                oriented_view = sprite_params[5],
                remembers = sprite_params[6],
                walls = sprite_params[7],
            )
        vgdl.registry.register(sprite.__name__, sprite)
        env_name = register_vgdl_env(domainfile, builder.level_name, observer_cls, blocksize, env_counter)
        env_counter += 1
        env = gym.make(env_name)

        test_state_sequence = run_simulation(action_sequence, env)

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

    env.close()
    builder.close()


if __name__ == '__main__':
    main()
