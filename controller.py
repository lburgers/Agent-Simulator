import gym
from gym.envs.registration import register
import vgdl.interfaces.gym
from vgdl.util.humanplay.controls import VGDLControls
from vgdl.ontology.constants import *
from pygame.math import Vector2

import numpy as np
import itertools
import time
import uuid
import os
import copy
import shutil

import imageio
import png

from sprite import CustomAStarChaser
from build_level import BuildLevel

reverse_action = {
	0: 1,
	1: 0,
	2: 3,
	3: 2,
	4: 4,
}

class Controller():

	def __init__(self, position_object, sprite_params,  pref=None, policy_file=None):

		self.positions = position_object

		self.true_sprite_params = sprite_params
		self.level_file = './level.txt'
		self.domain_file = './game.txt'
		self.observer_cls = 'objects'
		self.blocksize = 24
		self.env_counter = 0
		self.prefix = pref

		self.trial_count = 10

		self.env = None
		self.builder = None

		self.build_map(position_object)

		self.policies = None
		if policy_file:
			self.policies = np.load(policy_file)['arr_0']


	def build_map(self, position_object):

		self.builder = BuildLevel(self.level_file)

		for key in position_object.keys():
			cords = position_object[key]
			self.builder.add(cords[0], cords[1], key)

		self.builder.save()


	def test_sequence(self, action_sequence, true_sequence, debug=False):

		prob = 0.0
		sprite = None
		old_pos = None
		old_diff = None

		self.env.reset()

		# TODO: add direction change probabilities

		for step_i in itertools.count():

			prob_multiplier = 0.0


			if sprite and sprite.searching:

				next_cords = (true_sequence[step_i][0], true_sequence[step_i][1])
				old_dict = copy.deepcopy(sprite.__dict__)
				obs, reward, done, sprite = self.env.step(action_sequence[step_i], next_cords)
				current_pos = (sprite.rect[0], sprite.rect[1])

				if not sprite.searching:
					self.env.step(reverse_action[action_sequence[step_i]])
					sprite.positionUpdate((true_sequence[step_i-1][0], true_sequence[step_i-1][1]))
					sprite.set_dict(old_dict) # bring sprite back one time step

					obs, reward, done, sprite = self.env.step(action_sequence[step_i]) # redo step


				elif sprite.velocity == Vector2(0,0):
					if debug: print('stationary 0')
					return 0.0
				else:
					diff = (current_pos[0] - old_pos[0], current_pos[1] - old_pos[1])
					if old_diff and diff != old_diff:
						prob_multiplier = np.log(1/3.0)
						if debug: print('1/3')

					old_diff = diff


			else:

				obs, reward, done, sprite = self.env.step(action_sequence[step_i])
				if sprite.searching and not done:
					sprite.positionUpdate((true_sequence[step_i][0], true_sequence[step_i][1]))
					prob += np.log(1/3.0)
					old_pos = (obs[0], obs[1])
					if debug: print('1/3')
					continue
			
			old_pos = (obs[0], obs[1])

			if done:
				break

			if (true_sequence[step_i] == obs).all():
				prob += prob_multiplier
			else:
				if debug: print('missmatch 0')
				return 0.0

			if step_i == len(action_sequence) - 1:
				break

		if debug: print('match ' + str(np.exp(prob)))
		return np.exp(prob)


		
	def run_simulation(self, action_sequence, human=False, save=False):
		if save:
			unique_filename = str(uuid.uuid4())
			basedir = './trials/%s/' % unique_filename
			if os.path.exists(basedir):
				shutil.rmtree(basedir) 
			os.mkdir(basedir)

		if human:
			controls = VGDLControls(self.env.unwrapped.get_action_meanings())
			self.env.render('human')

		states = np.array([])
		self.env.reset()
		actions_used = []
		for step_i in itertools.count():

			if human:
				controls.capture_key_presses()
				action = controls.current_action
				actions_used.append(action)
				obs, reward, done, info = self.env.step(action)
				self.env.render()
			else:
				obs, reward, done, info = self.env.step(action_sequence[step_i])
				actions_used.append(action_sequence[step_i])

			states = [obs] if len(states) == 0 else np.vstack((states, obs))

			if save:
				rgb_array = self.env.render('rgb_array')
				rgb_shape = rgb_array.shape
				png.from_array(rgb_array.reshape(-1, 3*rgb_shape[1]), 'RGB').save("%s%d.png"% (basedir, step_i))

			if done:
				break
			if not human and step_i == len(action_sequence) - 1:
				break

			if human:
				time.sleep( 1/ 15.0)

		if save:
			images = []
			for i in range(len(os.listdir(basedir))-1):
				images.append(imageio.imread(basedir + '%d.png' % (i)))
			imageio.mimsave('./trials/%s.gif' % unique_filename, images)
			shutil.rmtree(basedir)

			self.save_log_file(unique_filename, actions_used)

		return states, actions_used

	def save_log_file(self, unique_filename, action_sequence):
		grid_string = self.builder.grid_string()
		action_string = str(action_sequence)
		true_sprite_string = str(self.true_sprite_params)

		full_file = '%s \n\n %s \n\n %s \n\n %s \n' % (grid_string, self.positions, true_sprite_string, action_string)

		with open('./trials/%s.txt' % unique_filename, 'w') as f:
			f.write(full_file)


	def make_sprite(self, sprite_params, route, direction, home):

		class CustomNPC(CustomAStarChaser):

			lost_function = sprite_params[0]
			tom = sprite_params[1]
			memory = sprite_params[2]
			forgets = sprite_params[3]
			hearing = sprite_params[4]
			orientation = direction

			# reset
			mode = 'DEFENSIVE'
			searching = False
			alert_step = 0
			initial_orientation = None
			last_player_cords = None # last target
			current_target = None
			player_desire_cords = None
			home_cords = home
			static_route = route
			static_route_index = 0
			policies = self.policies

		return CustomNPC


	def make_env(self, sprite_params, route=[], dir=LEFT, home=None):
		self.sprite = self.make_sprite(sprite_params, route, dir, home)
		vgdl.registry.register(self.sprite.__name__, self.sprite)
		env_name = self.register_vgdl_env()
		
		self.env = gym.make(env_name)


	def register_vgdl_env(self):
		env_name = 'vgdl_{}-v0'.format(self.env_counter)
		if self.prefix:
			env_name = '{}-vgdl_{}-v0'.format(self.prefix, self.env_counter)

		register(
			id=env_name,
			entry_point='vgdl.interfaces.gym:VGDLEnv',
			kwargs={
				'game_file': self.domain_file,
				'level_file': self.builder.level_name,
				'block_size': self.blocksize,
				'obs_type': self.observer_cls,
			},
		)

		self.env_counter += 1

		return env_name

	def close(self):
		# self.env.close()
		self.builder.close()
