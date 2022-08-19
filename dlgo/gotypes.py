## this file defines the classes that are used to facilitate the playing of
# a game of go, such as the player class

## File imports
import enum
from collections import namedtuple


## This is the class definition of a player, used to track the turn order
class Player(enum.Enum):
    black = 1
    white = 2

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white


## This class represents a point on the board
# namedtuples are used instead of normal tuples as they allow the calling
# of 'row' and 'col' which is more intuitive than Point[0] or Point[1] and
# helps with the readability.
class Point(namedtuple('Point', 'row col')):
    def neighbors(self):
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]

