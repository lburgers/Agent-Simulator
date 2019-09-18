
import numpy as np
import pygame
from pygame.math import Vector2

from vgdl.core import VGDLSprite, Action, Resource, Immutable
from vgdl.ai import AStarWorld
from vgdl.ontology.sprites import RandomNPC
from vgdl.ontology.constants import *


class CustomAStarChaser(RandomNPC):
    """ Move towards the character using A* search. """
    stype = None

    speed = 1

    tom = True

    search = True
    target = 'avatar'
    stationary = False
    random = False
    fleeing = False
    lost_function = 'random'

    see_through_walls = False
    sight_limit = 20
    full_field_view = False
    orientation = LEFT
    
    memory = True
    old_target = None
    current_target = None

    # save avatar goal locations
    avatar_goals = {}

    def getWallDistances(self, world):
        wall_dists = [self.speed for _ in BASEDIRS]
        tileX, tileY = self.rect.x, self.rect.y

        tileSet = [[(tileX, tiley) for tiley in range(int(tileY-1), int(tileY-self.speed-1), -1)]]
        tileSet.append([(tilex, tileY) for tilex in range(int(tileX-1), int(tileX-self.speed-1), -1)])
        tileSet.append([(tileX, tiley) for tiley in range(int(tileY+1), int(tileY+self.speed+1))])
        tileSet.append([(tilex, tileY) for tilex in range(int(tileX+1), int(tileX+self.speed+1))])

        for i in range(len(tileSet)):
            for (tilex, tiley) in tileSet[i]:
                index = world.get_index(tilex, tiley)
                if index in world.wall_tile_indices:
                    diff = abs(tileY-tiley) + abs(tileX-tilex)
                    wall_dists[i] = diff - 1
                    break
        return wall_dists

    def randomUpdate(self, game):
        action = game.random_generator.choice(BASEDIRS)
        direction = BASEDIRS.index(action)
        wall_dists = self.getWallDistances(self.world)
        self.physics.active_movement(self, action, wall_dists[direction])

    def searchUpdate(self, game, goal_cords):
        goal_sprite = None
        max_distance = float('-Inf')

        for s in game.sprite_registry.sprites():
            # handle avoidance (go to opposite side of map from target)
            if self.fleeing:
                dist = (abs(s.rect.y - goal_cords[1]) + abs(s.rect.x - goal_cords[0]))
                index = self.world.get_index(s.rect.x, s.rect.y)
                # find furthest tile from target which is not a wall
                if index not in self.world.wall_tile_indices and dist >= max_distance:
                    max_distance = dist
                    goal_sprite = s
            else:
                if goal_cords[0] == s.rect.x and goal_cords[1] == s.rect.y:
                    goal_sprite = s

        path = self.world.getMoveFor(self, goal_sprite)

        if path and len(path)>1:
            move = path[1]

            nextX, nextY = self.world.get_sprite_tile_position(move.sprite)
            nowX, nowY = self.world.get_sprite_tile_position(self)

            diffX = abs(nextX - nowX)
            diffY = abs(nextY - nowY)

            if nowX == nextX:
                if nextY > nowY:
                    movement = DOWN
                else:
                    movement = UP
            else:
                if nextX > nowX:
                    movement = RIGHT
                else:
                    movement = LEFT
            
            self.orientation = movement
            self._update_position(movement, speed=diffX+diffY)

    def _boundedCords(self, game, x, y):
        bounded_x, bounded_y = game.width, game.height
        if y < 0: bounded_y = 0
        if x < 0: bounded_x = 0
        if x >= 0 and x < game.width:
            bounded_x = x
        if y >= 0 and y < game.height:
            bounded_y = y
        return bounded_x, bounded_y

    def addWalls(self, game, matrix):
        new_matrix = matrix
        perceiverX, perceiverY = self.rect.x, self.rect.y

        for x in range(game.width):
            for y in range(game.height):
                index = self.world.get_index(x, y)
                if index in self.world.wall_tile_indices:
                    if x <= perceiverX and y <= perceiverY:
                        new_matrix[:x+1, :y+1] = 0
                    if x >= perceiverX and y <= perceiverY:
                        new_matrix[x:, :y+1] = 0
                    if x >= perceiverX and y >= perceiverY:
                        new_matrix[x:, y:] = 0
                    if x <= perceiverX and y >= perceiverY:
                        new_matrix[:x+1, y:] = 0
        return new_matrix


    def buildPerceptionMatrix(self, game):
        matrix = np.zeros((game.width, game.height))
        x, y = self.rect.x, self.rect.y

        # add sight limit (default)
        aboveCords = self._boundedCords(game, x+self.sight_limit+1, y+self.sight_limit+1)
        belowCords = self._boundedCords(game, x-self.sight_limit, y-self.sight_limit)
        matrix[ belowCords[0] : aboveCords[0], belowCords[1] : aboveCords[1] ] = 1

        if not self.full_field_view:
            if self.orientation == UP:
                matrix[:, x+self.speed+1:] = 0
            elif self.orientation == DOWN:
                matrix[:, :y-self.speed] = 0
            elif self.orientation == LEFT:
                matrix[x+self.speed+1:, :] = 0
            elif self.orientation == RIGHT:
                matrix[:x-self.speed, :] = 0


        # handle walls blocking vision
        if not self.see_through_walls:
            matrix = self.addWalls(game, matrix)

        return matrix

    def print_matrix(self, matrix):
        for i in range(matrix.shape[1]):
            print(matrix[:,i])

    def add_avatar_goals(self, game):
        if self.avatar_goals == {}:
            for goal in ['A', 'B', 'C']:
                avatar_list = game.get_sprites(goal)
                if len(avatar_list) == 1:
                    position = avatar_list[0].rect
                    self.avatar_goals[goal] = (position[0], position[1])

    def distance(self, r1, r2):
        """ Grid physics use Hamming distances. """
        return (abs(r1[1] - r2[1])
                + abs(r1[0] - r2[0]))

    def infer_goal(self, game):
        best_goal = None
        min_change = float('inf')

        goal_dists = {}

        goals = [key for key in self.avatar_goals.keys()]
        for goal in goals:
            position = self.avatar_goals[goal]
            old_dist = self.distance((self.last_player_cors[0],self.last_player_cors[1]), position)
            new_dist = self.distance((self.player_sprite.rect[0],self.player_sprite.rect[1]), position)

            goal_dists[goal] = new_dist

            dist_chang = new_dist - old_dist
            # import pdb; pdb.set_trace()
            if dist_chang < min_change or best_goal == None:
                best_goal = goal
                min_change = dist_chang
            elif dist_chang == min_change and new_dist < goal_dists[best_goal]:
                best_goal = goal
                min_change = dist_chang


        return self.avatar_goals[best_goal]


    def update(self, game):

        self.player_sprite = game.get_sprites(self.target)[0]
        player_x, player_y = self.player_sprite.rect.x, self.player_sprite.rect.y

        self.add_avatar_goals(game)

        self.world = AStarWorld(game, self.speed)

        if self.stationary:
            return
        elif self.random:
            self.randomUpdate(game)

        elif self.search:

            perception_matrix = self.buildPerceptionMatrix(game)
            # self.print_matrix(perception_matrix)

            # if the target is in view
            if perception_matrix[player_x, player_y] == 1:

                self.current_target = (player_x, player_y)

                # infer desire if in view and has memory
                if self.tom and self.old_target and self.memory:
                    desire_cords = self.infer_goal(game)
                    # self.current_target = desire_cords

                    # TODO: add intercept

                self.old_target = self.current_target

                self.searchUpdate(game, self.current_target)

            
            # if not in view
            else:
                # if can't see target and real target forget
                # if self.current_target and perception_matrix[self.current_target[0], self.current_target[1]] == 1:
                #     self.current_target = None

                # if it remembers last seen cords and has memory go there
                if self.current_target and self.memory:
                    self.searchUpdate(game, self.current_target)
                
                # else fully lost
                else:
                    if self.lost_function == 'random':
                        self.randomUpdate(game)

        self.last_player_cors = self.player_sprite.rect.x, self.player_sprite.rect.y


