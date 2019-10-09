import gym
from gym.envs.registration import register
import vgdl.interfaces.gym
from vgdl.util.humanplay.controls import VGDLControls
from vgdl.ontology.constants import *

import numpy as np
import itertools
import time
import uuid
import os
import shutil

import imageio
import png

from sprite import CustomAStarChaser
from build_level import BuildLevel


class Controller():

	def __init__(self, position_object, sprite_params,  pref=None):

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

	def build_map(self, position_object):

		self.builder = BuildLevel(self.level_file)

		for key in position_object.keys():
			cords = position_object[key]
			self.builder.add(cords[0], cords[1], key)

		self.builder.save()

		
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
		human_actions = []
		for step_i in itertools.count():

			if human:
				controls.capture_key_presses()
				action = controls.current_action
				human_actions.append(action)
				obs, reward, done, info = self.env.step(action)
				self.env.render()
			else:
				obs, reward, done, info = self.env.step(action_sequence[step_i])

			if save:
				rgb_array = self.env.render('rgb_array')
				rgb_shape = rgb_array.shape
				png.from_array(rgb_array.reshape(-1, 3*rgb_shape[1]), 'RGB').save("%s%d.png"% (basedir, step_i))

			if done:
				break
			if not human and step_i == len(action_sequence) - 1:
				break

			states = [obs] if len(states) == 0 else np.vstack((states, obs))

			if human:
				time.sleep( 1/ 15.0)

		if save:
			images = []
			for i in range(len(os.listdir(basedir))-1):
				images.append(imageio.imread(basedir + '%d.png' % (i)))
			imageio.mimsave('./trials/%s.gif' % unique_filename, images)
			shutil.rmtree(basedir)

			if human:
				self.save_log_file(unique_filename, human_actions)
			else:
				self.save_log_file(unique_filename, action_sequence)

		if human:
			return states, human_actions
		else:
			return states, action_sequence

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
