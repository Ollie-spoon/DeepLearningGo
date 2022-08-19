## This file

## Imports
from dlgo.gotypes import Point


## Verify that a point is an eye or not using the rules:
# 1. All neighbors must be off the board or the same colour
# 2. Of the diagonal points or 'corners', 3 or 4 of these must
# be the same colour, unless at least one corner is not on the
# board, then it must be all available corners.
def is_point_an_eye(board, point, color):

    ## Point has to be empty to be an eye
    if board.get(point) is not None:
        return False

    ## All neighbors must be the same colour, return False
    # otherwise
    for neighbor in point.neighbors():
        if board.is_on_grid(neighbor):
            neighbor_color = board.get(neighbor)
            if neighbor_color != color:
                return False

    ## count corners that are the same colour and off the board
    friendly_corners = 0
    off_board_corners = 0
    corners = [
        Point(point.row - 1, point.col - 1),
        Point(point.row - 1, point.col + 1),
        Point(point.row + 1, point.col - 1),
        Point(point.row + 1, point.col + 1),
    ]
    for corner in corners:
        if board.is_on_grid(corner):
            corner_color = board.get(corner)
            if corner_color == color:
                friendly_corners += 1
        else:
            off_board_corners += 1

    ## if some corners are off the board then requires all corners
    # to be the same colour. Otherwise, 3 or 4 same coloured corners
    # are both acceptable
    if off_board_corners > 0:
        return off_board_corners + friendly_corners == 4
    return friendly_corners >= 3

