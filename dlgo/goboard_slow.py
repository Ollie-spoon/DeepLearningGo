## This file

## Imports
import copy
from dlgo.gotypes import Player


## This class is used to either play a piece, pass a turn, or resign.
# The word 'move' refers to the combination of these three options.

# This will not be called directly, instead Move.play or move.resign
# will be used
class Move():

    ## Ensure that only one of the three options is picked and then set
    # each of the 'self.is_' to true or false based on which option is picked
    def __init__(self, point=None, is_pass=False, is_resign=False):
        assert (point is not None) ^ is_pass ^ is_resign  # ^ is XOR
        self.point = point
        self.is_play = (point is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign

    ## "Class methods are for when you need to have methods that aren't specific
    # to any particular instance, but still involve the class in some way."

    ## These three class methods are used to achieve, one of the three possible moves
    @classmethod
    def play(cls, point):
        return Move(point=point)

    @classmethod
    def pass_turn(cls):
        return Move(is_pass=True)

    @classmethod
    def resign(cls):
        return Move(is_resign=True)


## A 'string' in go is a connected group of stones. This class is used to track the
# number of liberties, stones, and the colour of a string.
class GoString():

    ## uses a set of stones and liberties to remove any duplicates
    def __init__(self, color, stones, liberties):
        self.color = color
        self.stones = set(stones)
        self.liberties = set(liberties)

    ## Removes a point from the set of liberties
    def remove_liberty(self, point):
        self.liberties.remove(point)

    ## Adds a point to the set of liberties
    def add_liberty(self, point):
        self.liberties.add(point)

    ## checks that two strings are of the same colour before creating a union of
    # the two sets of stones. The union operator '|' does not allow duplicates.
    # A new GoString is then created with the set of the combined stones, removing
    # the combined stones from the union of the liberties of the two sets as these
    # points are no longer liberties.
    def merged_with(self, go_string):
        assert go_string.color == self.color
        combined_stones = self.stones | go_string.stones
        return GoString(
            self.color,
            combined_stones,
            (self.liberties | go_string.liberties) - combined_stones)

    ## This property returns the number of liberties of a string
    @property
    def num_liberties(self):
        return len(self.liberties)

    ## Checks whether two strings are the same. '__eq__' is the defining function
    # for '=='. isinstance verifies that an object is of a specified type. '\'
    # extends the current logical line over into the next physical line of code.
    def __eq__(self, other):
        return isinstance(other, GoString) and \
               self.color == other.color and \
               self.stones == other.stones and \
               self.liberties == other.liberties


## The Board class contains a representation of the board, including the
# functionality to place stones
class Board():

    ## Initialises as empty with a set size
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {}

    ## Places a stone on the board.
    # This function also updates strings and their liberties if they have been
    # modified by the placing of a stone
    def place_stone(self, player, point):
        assert self.is_on_grid(point)
        assert self._grid.get(point) is None
        adjacent_same_color = []
        adjacent_opposite_color = []
        liberties = []

        ## Iterate through neighbors
        for neighbor in point.neighbors():

            ## if not on the board, ignore
            if not self.is_on_grid(neighbor):
                continue
            neighbor_string = self._grid.get(neighbor)

            ## if the neighbor is not in a string then it is a liberty
            if neighbor_string is None:
                liberties.append(neighbor)

            ## if it's in a string of the same colour then make sure it's in
            # adjacent_same_colour
            elif neighbor_string.color == player:
                if neighbor_string not in adjacent_same_color:
                    adjacent_same_color.append(neighbor_string)

            ## if it's in a string of the opposite colour then make sure it's
            # in adjacent_opposite_colour
            else:
                if neighbor_string not in adjacent_opposite_color:
                    adjacent_opposite_color.append(neighbor_string)

        ## Create a new string with this point in
        new_string = GoString(player, [point], liberties)

        ## Merge this string with adjacent strings of the same colour
        for same_color_string in adjacent_same_color:
            new_string = new_string.merged_with(same_color_string)

        ## Update the grid that this new string is at each point in the string
        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string

        ## Remove liberties from neighboring opposite coloured strings for
        # this point
        for other_color_string in adjacent_opposite_color:
            other_color_string.remove_liberty(point)

        ## If neighboring opposite coloured strings have 0 liberties then
        # remove them
        for other_color_string in adjacent_opposite_color:
            if other_color_string.num_liberties == 0:
                self._remove_string(other_color_string)

    ## Returns True if in the grid, False otherwise
    def is_on_grid(self, point):
        return 1 <= point.row <= self.num_rows and \
               1 <= point.col <= self.num_cols

    ## returns colour of stone at point (or None)
    def get(self, point):
        string = self._grid.get(point)
        return None if string is None else string.color

    ## returns a the GoString object at a point (or None)
    def get_go_string(self, point):
        string = self._grid.get(point)
        return string

    ## removes each point in a string, checks if the neighbors of each point
    # are in another string and adds a liberty to them if they are
    def _remove_string(self, string):
        for point in string.stones:
            for neighbor in point.neighbors():
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string:
                    neighbor_string.add_liberty(point)
            self._grid[point] = None


## A game state class used to ensure illegal moves aren't played.
# Note: previous is the previous GameState. This means that the entire game
# up to a move is accessible from the GameState at that move
class GameState():
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move

    ## records the new state of the board after a move has been done and
    # returns a new board state with this in
    def apply_move(self, move):
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board
        return GameState(next_board, self.next_player.other, self, move)

    ## Game is over if the player resigned or if the last two moves were passes
    def is_over(self):
        if self.last_move is None:
            return False
        if self.last_move.is_resign:
            return True
        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False
        return self.last_move.is_pass and second_last_move.is_pass

    ## Creates a copy of the board and then tries the proposed move and checks
    # the number of liberties of the string at the point played
    def is_move_self_capture(self, player, move):
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        new_string = next_board.get_go_string(move.point)
        return new_string.num_liberties == 0

    ## is a move ko (not properly understood)
    def does_move_violate_ko(self, player, move):
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board)
        past_state = self.previous_state
        while past_state is not None:
            if past_state.situation == next_situation:
                return True
            past_state = past_state.previous_state
        return False

    ## Checks if the game is over, if not then was a stone played,
    # if true then check:
    # 1. is that board point empty
    # 2. is the move a self capture
    # 3. does the move violate the ko rule
    def is_valid_move(self, move):
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True
        return (
                self.board.get(move.point) is None and
                not self.is_move_self_capture(self.next_player, move) and
                not self.does_move_violate_ko(self.next_player, move))

    ## This class method will instantiate a new game
    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)

    @property
    def situation(self):
        return (self.next_player, self.board)




