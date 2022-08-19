## this file contains utility functions that are helpful throughout

## Imports
from dlgo import gotypes

## Letters for the side of the board (wihtout I)
COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: ' . ',
    gotypes.Player.black: ' x ',
    gotypes.Player.white: ' o ',
}


## Output function
def print_move(player, move):
    if move.is_pass:
        move_str = 'passes'
    elif move.is_resign:
        move_str = 'resigns'
    else:
        move_str = '%s%d' % (COLS[move.point.col - 1], move.point.row)
    print('%s %s' % (player, move_str))


## Board print function
def print_board(board):
    for row in range(board.num_rows, 0, -1):
        bump = " " if row <= 9 else ""
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(gotypes.Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%s%d %s' % (bump, row, ''.join(line)))
    print('\t' + '  '.join(COLS[:board.num_cols]))


## used to input (F, 15) and output Point(6, 15)
def point_from_coords(coords):
    col = COLS.index(coords[0]) + 1
    row = int(coords[1:])
    return gotypes.Point(row=row, col=col)


## used to input Point(6, 15) and output (F, 15)
def coords_from_point(point):
    return '%s%d' % (
        COLS[point.col - 1],
        point.row
    )


##
def human_move_selection(human_move, goboard):
    if human_move.strip().lower() == "pass":
        move = goboard.Move.pass_turn()
    elif human_move.strip().lower() == "resign":
        move = goboard.Move.resign()
    else:
        point = point_from_coords(human_move.strip().upper())
        move = goboard.Move.play(point)
    return move
