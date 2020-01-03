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


	def calculate_prob(self, old_pos, sprite):
		actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
		game_width = 24

		conv_goal_cords = int((sprite.goal_cords[1])*game_width + sprite.goal_cords[0])
		conv_old_cords = int((old_pos[1])*game_width + old_pos[0])

		action = (sprite.rect.x - old_pos[0], sprite.rect.y - old_pos[1])
		if action == (0,0):
			return 1.0

		action_idx = actions.index(action)

		prob = self.policies[conv_goal_cords, action_idx, conv_old_cords]
		return prob

	def test_sequence(self, action_sequence, true_sequence, debug=False):
		log_prob = 0.0
		sprite = None
		old_pos = (true_sequence[0][0], true_sequence[0][1])
		old_diff = None

		states = ['searching', 'chasing', 'intercepting', 'patrolling', 'returning', 'waiting']
		state_record = np.zeros((len(true_sequence), len(states)))

		self.env.reset()

		for step_i in itertools.count():

			next_cords = (true_sequence[step_i][0], true_sequence[step_i][1])
			obs, reward, done, sprite = self.env.step(action_sequence[step_i], next_cords)

			state_idx = states.index(sprite.state)
			state_record[step_i, state_idx] = 1

			if done:
				break

			if sprite.searching:
				if sprite.velocity == Vector2(0,0):
					if debug: print('stationary 0.0')
					return 0.0, None
				else:
					diff = (obs[0] - old_pos[0], obs[1] - old_pos[1])
					if old_diff and diff != old_diff:
						log_prob += np.log(1/3.0)
						if debug: print('1/3')
					else:
						log_prob += 0.0

					old_diff = diff


			else:
				# if staying in the same place and no goal => perfect match
				if sprite.goal_cords == None and (step_i == 0 or \
									(true_sequence[step_i-1][:2] == true_sequence[step_i][:2]).all()):

					if debug: print('1.0')
					log_prob += 0.0

				# if moving with no goal => mismatch and break
				elif sprite.goal_cords == None:
					if debug: print('missmatch_ 0')
					return 0.0, None

				# if sprite not moving but has goal elsewhere => mismatch break
				elif sprite.goal_cords != (true_sequence[step_i][0], true_sequence[step_i][1]) and (step_i > 0 and \
									(true_sequence[step_i-1][:2] == true_sequence[step_i][:2]).all()):

					if debug: print('missmatch* 0')
					return 0.0, None

				else:
					# calculate mdp prob
					log_prob += np.log(self.calculate_prob(old_pos, sprite))
					if debug: print('using mdp prob', sprite.goal_cords, np.log(self.calculate_prob(old_pos, sprite)))

			if step_i == len(true_sequence) - 1:
				break

			old_pos = (obs[0], obs[1])
			if debug: print('updating old pos', old_pos)

		if debug: print('match ' + str(np.exp(log_prob)))
		return np.exp(log_prob), state_record


		
	def run_simulation(self, action_sequence, human=False, save=False, save_folder_path=None):
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
				obs, reward, done, sprite = self.env.step(action_sequence[step_i])
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
			self.convert_gif_to_mp4(unique_filename, states)
			shutil.rmtree(basedir)

			self.save_log_file(unique_filename, actions_used, states)

			if save_folder_path:
				self.move_files_to_folder(unique_filename, save_folder_path)

		return states, actions_used

	def convert_gif_to_mp4(self, unique_filename, states):
		fps = 10
		os.system('ffmpeg -i ./trials/%s.gif -movflags faststart -pix_fmt yuv420p -vf \"fps=10,scale=trunc(iw/2)*2:trunc(ih/2)*2\" ./trials/video.mp4' % unique_filename)
		os.system('ffmpeg  -i ./trials/video.mp4 -stream_loop -1 -i ./footstep.wav -shortest -map 0:v:0 -map 1:a:0 -y ./trials/video1.mp4')


		mute_start = None
		mute_sections = []
		last_state = states[0]
		for i, state in enumerate(states):
			if (last_state[2:] == state[2:]).all() and mute_start == None:
				mute_start = i
			if not (last_state[2:] == state[2:]).all() and mute_start != None:
				mute_sections.append( (mute_start, i) )
				mute_start = None
			last_state = state

		volume_string = "volume=enable=\'between(t,%.2f,%.2f)\':volume=0"
		mute_sections = [volume_string % (m[0]/fps, m[1]/fps) for m in mute_sections]
		os.system('ffmpeg -i ./trials/video1.mp4 -af \"%s\" -y ./trials/video_w_sound.mp4' % ','.join(mute_sections))

		os.system('rm ./trials/video.mp4')
		os.system('rm ./trials/video1.mp4')

	def move_files_to_folder(self, unique_filename, save_folder_path):
		os.system('mv ./trials/%s.gif %s' % (unique_filename, save_folder_path))
		os.system('mv ./trials/%s.txt %s' % (unique_filename, save_folder_path))
		os.system('mv ./trials/video_w_sound.mp4 %s' % (save_folder_path))

	def save_log_file(self, unique_filename, action_sequence, states):
		grid_string = self.builder.grid_string()
		action_string = str(action_sequence)
		true_sprite_string = str(self.true_sprite_params)

		full_file = '%s \n\n %s \n\n %s \n\n %s \n\n %s \n' % (grid_string,
															   self.positions,
															   true_sprite_string,
															   action_string,
															   states)

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
