## This file

__all__ = ['Agent']


## This currently is the base Agent class to be built upon
class Agent:
    def __init__(self):
        pass

    def select_move(self, game_state):
        raise NotImplementedError()
