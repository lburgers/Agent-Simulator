import glob
import re
import numpy as np


class Loader():

	def __init__(self, trial):

		content = self.load_trial_contents(trial)
		lines = content.split('\n\n')
		self.locations = self.parse_location(lines[1])
		self.true_params = self.parse_params(lines[2])
		self.player_actions = self.parse_actions(lines[3])

		self.state_sequence = None
		if len(lines) > 4:
			self.state_sequence = self.parse_states(lines[4])

	def load_trial_contents(self, trial):
	    gif_filename = glob.glob('./trials/%s/*.gif' % trial)[0]
	    trial_uuid = gif_filename.split('.gif')[0]

	    with open(trial_uuid + '.txt', "r") as f:
	        content = f.read()
	    return content

	def parse_location(self, line):
		locations = {'A': None, '0': None, 'X': None}
		for key in locations.keys():
			try:
				x_re = '\'%s\': \((.*?),' % key
				y_re = "\'%s\': \([0-9]+, (.*?)\)" % key
				x = int( re.findall(x_re, line)[0] )
				y = int( re.findall(y_re, line)[0] )
				locations[key] = (x, y)
			except:
				pass

		return locations

	def parse_actions(self, line):
		sequence = re.findall(r'\[(.*)\]', line)
		sequence = sequence[0].split(', ')
		sequence = [int(a) for a in sequence]
		return sequence

	def parse_params(self, line):
		params = re.findall(r'\((.*)\)', line)[0]
		params = params.split(', ')
		lost_func = re.findall(r'\'(.*)\'', params[0])[0]
		tom = True if params[1] == 'True' else False
		remembers = True if params[2] == 'True' else False
		forgets = True if params[3] == 'True' else False
		hears = True if params[4] == 'True' else False

		params = (lost_func, tom, remembers, forgets, hears)
		return params

	def parse_states(self, line):
		parsed_states = re.findall(r'\[([ 0-9].*?)\]', line)
		states = np.zeros((len(parsed_states), 4))

		for i in range(len(parsed_states)):
			parsed_state = parsed_states[i].split(' ')
			parsed_state = list(filter(None, parsed_state))
			parsed_state = np.array([int(s) for s in parsed_state])

			states[i, :] = parsed_state

		return states
