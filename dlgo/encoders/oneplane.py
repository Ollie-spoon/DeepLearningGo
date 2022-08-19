import numpy as np

from dlgo.encoders.base import Encoder
from dlgo.goboard_fast import Point


class OnePlaneEncoder(Encoder):
    def __init__(self, board_size):
        self.board_width, self.board_height = board_size
        self.num_planes = 1
    
    def name(self):
        return 'oneplane'
    
    def encode(self, game_state):  # <1>
        board_matrix = np.zeros(self.shape())
        next_player = game_state.next_player
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)
                if go_string is None:
                    continue
                if go_string.color == next_player:
                    board_matrix[0, r, c] = 1
                else:
                    board_matrix[0, r, c] = -1
        return board_matrix
    
    def encode_point(self, point):  # <2>
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):  # <3>
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)
    
    def num_points(self):
        return self.board_width * self.board_height
    
    def shape(self):
        return self.num_planes, self.board_height, self.board_width


def create(board_size):
    return OnePlaneEncoder(board_size)


"""
<1> takes the current game_state and for each row, col
        if that point is in a string: 
            set it to 1 if it's the same as the bot
            set it to -1 if it's the opposition
        otherwise set it to 0

<2> turn a instance of a Point into an integer position index

<3> turn an integer position index into a Point
"""