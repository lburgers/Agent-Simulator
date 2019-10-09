
import numpy as np
import random
import pygame
from pygame.math import Vector2

from vgdl.core import VGDLSprite, Action, Resource, Immutable
from vgdl.ai import AStarWorld
from vgdl.ontology.sprites import RandomNPC
from vgdl.ontology.constants import *

DEFENSIVE = "DEFENSIVE"
ALERT = "ALERT"

class CustomAStarChaser(RandomNPC):
    """ Move towards the character using A* search. """
    stype = None

    # parameters
    lost_function = None
    tom = True
    memory = True
    forgets = True
    hearing = True
    orientation = None

    # default
    speed = 1
    sight_limit = 25
    memory_limit = 40
    hearing_limit = 5
    target = 'avatar'
    fleeing = False

    # utilities for tracking targets
    mode = DEFENSIVE
    searching = False
    alert_step = 0
    initial_orientation = None
    home_cords = None
    last_player_cords = None # last target
    current_target = None
    avatar_goals = {} # avatar goal locations
    player_desire_cords = None
    corners = []
    static_route = []
    static_route_index = 0

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

    def AstarPath(self, game, start_sprite, goal_cords):
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

        path = self.world.getMoveFor(start_sprite, goal_sprite)

        if path == None:
            return []
        else:
            next_cords = [self.world.get_sprite_tile_position(move.sprite) for move in path]

            return next_cords

    def searchUpdate(self, game, goal_cords):

        path = self.AstarPath(game, self, goal_cords)

        if path and len(path)>1:

            nextX, nextY = path[1]
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

    def findCorners(self, game):
        def checkCorner(corner):
            if corner[0] >= 0 and corner[0] < game.width \
                and corner[1] >= 0 and corner[1] < game.height \
                and corner not in self.corners:
                return True
            return False

        for index in self.world.wall_tile_indices:
            x = index % game.width
            y = (index - x) / game.width
            above_index = self.world.get_index(x, y-1)
            below_index = self.world.get_index(x, y+1)
            left_index = self.world.get_index(x-1, y)
            right_index = self.world.get_index(x+1, y)

            if below_index in self.world.wall_tile_indices and right_index in self.world.wall_tile_indices:
                if checkCorner((x + 1, y + 1 )): self.corners.append( (x + 1, y + 1 ) )
            if above_index in self.world.wall_tile_indices and right_index in self.world.wall_tile_indices:
                if checkCorner((x + 1, y - 1 )): self.corners.append( (x + 1, y - 1 ) )
            if below_index in self.world.wall_tile_indices and left_index in self.world.wall_tile_indices:
                if checkCorner((x - 1, y + 1 )): self.corners.append( (x - 1, y + 1 ) )
            if above_index in self.world.wall_tile_indices and left_index in self.world.wall_tile_indices:
                if checkCorner((x - 1, y - 1 )): self.corners.append( (x - 1, y - 1 ) )

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

        if self.orientation == UP:
            matrix[:, y+self.speed+1:] = 0
        elif self.orientation == DOWN:
            matrix[:, :y-self.speed] = 0
        elif self.orientation == LEFT:
            matrix[x+self.speed+1:, :] = 0
        elif self.orientation == RIGHT:
            matrix[:x-self.speed, :] = 0


        # handle walls blocking vision
        matrix = self.addWalls(game, matrix)

        if self.hearing:
            # add hearing limit (default)
            aboveCords = self._boundedCords(game, x+self.hearing_limit+1, y+self.hearing_limit+1)
            belowCords = self._boundedCords(game, x-self.hearing_limit, y-self.hearing_limit)
            matrix[ belowCords[0] : aboveCords[0], belowCords[1] : aboveCords[1] ] = 1

        return matrix

    def print_matrix(self, matrix):
        old_value = matrix[self.rect.x, self.rect.y]
        matrix[self.rect.x, self.rect.y] = 2
        for i in range(matrix.shape[1]):
            print(matrix[:,i])
        matrix[self.rect.x, self.rect.y] = old_value

    def add_avatar_goals_and_home(self, game):
        if self.initial_orientation == None:
            self.initial_orientation = self.orientation
        if self.home_cords == None:
            self.home_cords = (self.rect.x, self.rect.y)
        if self.avatar_goals == {}:
            for goal in ['A', 'B', 'C']:
                avatar_list = game.get_sprites(goal)
                if len(avatar_list) == 1:
                    position = avatar_list[0].rect
                    self.avatar_goals[goal] = (position[0], position[1])

        # initialize static route
        if self.lost_function == 'route' and self.static_route == []:
            # number_of_points = np.random.randint(2, 5) # has to be 2 or greater to be a loop
            self.static_route_index = 0
            number_of_points = 1 # TODO: add way to control this
            self.findCorners(game)
            self.static_route.append(self.home_cords)
            for _ in range(number_of_points):
                # while True:
                #     rand_x = np.random.randint(0, game.width)
                #     rand_y = np.random.randint(0, game.height)
                #     index = self.world.get_index(rand_x, rand_y)
                #     if index not in self.world.wall_tile_indices:
                #         break

                self.static_route.append(random.choice(self.corners))

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
            old_dist = self.distance((self.last_player_cords[0],self.last_player_cords[1]), position)
            new_dist = self.distance((self.player_sprite.rect[0],self.player_sprite.rect[1]), position)

            goal_dists[goal] = new_dist

            dist_change = new_dist - old_dist

            if dist_change < min_change or best_goal == None:
                best_goal = goal
                min_change = dist_change
            elif dist_change == min_change and new_dist < goal_dists[best_goal]:
                best_goal = goal
                min_change = dist_change

        # if not approaching goal set desire to current position
        if (min_change >= 0):
            return (self.player_sprite.rect[0],self.player_sprite.rect[1])

        return self.avatar_goals[best_goal]

    def intercept_path(self, game, desire_cords):
        player_path = self.AstarPath(game, self.player_sprite, desire_cords)
        player_x, player_y = self.player_sprite.rect.x, self.player_sprite.rect.y
        min_length = float('Inf')
        min_path_diff = float('Inf')
        best_path_pos = None

        sprite_to_player = self.AstarPath(game, self, (player_x, player_y))
        to_player_len = len(sprite_to_player)

        for i in range(1, len(player_path)):
            path_pos = player_path[i]

            sprite_to_path = self.AstarPath(game, self, path_pos)
            sprite_path_len = len(sprite_to_path)

            path_len_diff = abs((i + 1) - sprite_path_len)

            if sprite_path_len <= min_length and path_len_diff == min_path_diff:
                min_length = sprite_path_len
                best_path_pos = path_pos
                min_path_diff = path_len_diff
            elif path_len_diff < min_path_diff:
                min_length = sprite_path_len
                best_path_pos = path_pos
                min_path_diff = path_len_diff

        if to_player_len-1 <= min_path_diff: return (player_x, player_y)
        if best_path_pos == None: return desire_cords
        return best_path_pos

    def update(self, game):

        self.player_sprite = game.get_sprites(self.target)[0]
        player_x, player_y = self.player_sprite.rect.x, self.player_sprite.rect.y

        self.world = AStarWorld(game, self.speed)
        self.findCorners(game)
        self.add_avatar_goals_and_home(game)

        perception_matrix = self.buildPerceptionMatrix(game)

        in_view = False

        # if the target is in view
        if perception_matrix[player_x, player_y] == 1:

            self.current_target = (player_x, player_y)

            # infer desire if in view and has memory
            if self.tom and self.last_player_cords and self.memory:
                self.player_desire_cords = self.infer_goal(game)
                intercept_pos = self.intercept_path(game, self.player_desire_cords)
                self.current_target = intercept_pos

            self.last_player_cords = (player_x, player_y)
            self.mode = ALERT
            self.alert_step = 0
            in_view = True
            self.searching = False


        if self.mode == ALERT:

            if in_view:
                self.searchUpdate(game, self.current_target)
            else:

                self.alert_step += 1
                if not self.memory or (self.forgets and self.alert_step > self.memory_limit):
                    self.mode = DEFENSIVE

                # if can't see real target and made it to current target => lost
                if self.current_target == (self.rect.x, self.rect.y):
                    self.current_target = None
                if self.player_desire_cords == (self.rect.x, self.rect.y):
                    self.player_desire_cords = None

                # if has current target and mem go to last location
                if self.current_target and self.memory:
                    self.searchUpdate(game, self.current_target)

                # if lost but knows which goal player was headed towards => go to their goal
                elif self.player_desire_cords and self.tom and self.memory:
                    self.searchUpdate(game, self.player_desire_cords)

                # if lost target, has mem, but no tom => start searching
                elif not self.current_target and self.memory:
                    self.searching = True
                    visible_indices = np.nonzero(perception_matrix)
                    next_index = np.random.choice(len(visible_indices[0]))

                    self.current_target = (visible_indices[0][next_index], visible_indices[1][next_index])
                    self.searchUpdate(game, self.current_target)


        elif self.mode == DEFENSIVE:

            self.searching = False

            if self.lost_function == 'home':
                # if home go to initial orientation
                if self.home_cords == (self.rect.x, self.rect.y):
                    self.orientation = self.initial_orientation

                self.searchUpdate(game, self.home_cords)
            
            elif self.lost_function == 'route':
                if (self.rect.x, self.rect.y) == self.static_route[self.static_route_index]:
                    self.static_route_index = (self.static_route_index + 1) % len(self.static_route)

                self.searchUpdate(game, self.static_route[self.static_route_index])
           
            elif self.lost_function == 'stationary':
                return
        
        


