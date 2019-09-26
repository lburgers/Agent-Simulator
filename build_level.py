import tempfile
import os

class BuildLevel:

	def __init__(self, level_file):
		self.level_name = 'tmp.txt'
		with open(level_file, 'r') as file:
		    content = file.read()
		    lines = content.split('\n')
		    self.grid = [list(line) for line in lines]

	def add(self, x, y, char):
		self.grid[y][x] = char

	def grid_string(self):
		grid_string = ''
		for row in self.grid:
			for c in row:
				grid_string += c
			grid_string += '\n'
		return grid_string

	def save(self):
		new_content = '\n'.join([''.join(row) for row in self.grid])
		tmp_file = open(self.level_name, "w")
		tmp_file.write(new_content)
		tmp_file.close()

	def close(self):
		os.remove(self.level_name)
