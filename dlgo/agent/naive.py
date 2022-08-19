## This file is the most naive version of a bot possible, one that will
# randomly select a move that doesn't fill in one of it's own eyes

import random
from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo.goboard_slow import Move
from dlgo.gotypes import Point


## This is a subclass of Agent representing the most basic 30 kyu bot
class RandomAgent(Agent):

    ## this is a redefinition of select_move that takes the game_state
    # and outputs a move
    def select_move(self, game_state):
        candidates = []
        for r in range(1, game_state.board.num_rows + 1):
            for c in range(1, game_state.board.num_cols + 1):
                candidate = Point(row=r, col=c)
                if game_state.is_valid_move(Move.play(candidate)) and not \
                        is_point_an_eye(game_state.board, candidate, game_state.next_player):
                    candidates.append(candidate)
        if not candidates:
            return Move.pass_turn()
        return Move.play(random.choice(candidates))
