## This file contains the monte carlo tree search algorithm

import math
import random

from dlgo.agent.naive_fast import FastRandomBot
from dlgo.agent.base import Agent
from dlgo.gotypes import Player
from dlgo.utils import coords_from_point
from dlgo.goboard_fast import Move

__all__ = [
    'MCTSAgent',
]


## fmt is a function to format text outputs.
# It take either a player or move, x and returns a readable string version
def fmt(x):
    if x is Player.black:
        return 'B'
    if x is Player.white:
        return 'W'
    if x.is_pass:
        return 'pass'
    if x.is_resign:
        return 'resign'
    return coords_from_point(x.point)


## show_tree outputs a visual representation of the current tree from a single node
def show_tree(node, indent='', max_depth=3):
    if max_depth < 0:
        return
    if node is None:
        return
    if node.parent is None:
        print('%sroot' % indent)
    else:
        player = node.parent.game_state.next_player
        move = node.move
        print('%s%s %s %d %.3f' % (
            indent, fmt(player), fmt(move),
            node.num_rollouts,
            node.winning_frac(player),
        ))
    for child in sorted(node.children, key=lambda n: n.num_rollouts, reverse=True):
        show_tree(child, indent + '  ', max_depth - 1)


## This is the class definition for a node object
# tag::mcts-node[]
class MCTSNode(object):
    """
    initial win_counts, rollouts, children are set to zero
    all legal moves = unvisited moves
    """
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {
            Player.black: 0,
            Player.white: 0,
        }
        self.num_rollouts = 0
        self.children = []
        self.unvisited_moves = game_state.legal_moves(is_resign=False)
# end::mcts-node[]

    """
    add_random_child takes an unvisited move, removes it and adds a child node
    containing that move. The child node is then returned 
    """
# tag::mcts-add-child[]
    def add_random_child(self):
        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        new_game_state = self.game_state.apply_move(new_move)
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node
# end::mcts-add-child[]

    """
    record_win adds a single count to both win_counts[winner] and num_rollouts
    """
# tag::mcts-record-win[]
    def record_win(self, winner):
        self.win_counts[winner] += 1
        self.num_rollouts += 1
# end::mcts-record-win[]

    """
    Three simple return functions
    """
# tag::mcts-readers[]
    def can_add_child(self):
        return len(self.unvisited_moves) > 0

    def is_terminal(self):
        return self.game_state.is_over()

    def winning_frac(self, player):
        return float(self.win_counts[player]) / float(self.num_rollouts)
# end::mcts-readers[]


## New subclass of the Agent class fro a MCTSAgent
class MCTSAgent(Agent):
    """
    class init of the agent.Agent class with self
    sets the temperature and the number of rounds
    rounds are defined as the number of MCTS iterations in a turn
    """
    def __init__(self, num_rounds, temperature, output = True):
        Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.output = output

    """
    Move selection:
    iterate(rounds)
      First walk down the tree until a new node can be added and add it
      then from this node simulate a random game and declare the winner 
      The win is then recorded for all nodes along this branch of the tree 
      all the way up
    Then create a list containing the scores for the root children
    return the move with the highest winning fraction.
    """
# tag::mcts-signature[]
    def select_move(self, game_state):
        root = MCTSNode(game_state)
# end::mcts-signature[]

# tag::mcts-rounds[]
        for i in range(self.num_rounds):
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            # Add a new child node into the tree.
            if node.can_add_child():
                node = node.add_random_child()

            # Simulate a random game from this node.
            winner = self.simulate_random_game(node.game_state)

            # Propagate scores back up the tree.
            while node is not None:
                node.record_win(winner)
                node = node.parent
# end::mcts-rounds[]

        if self.output:
            scored_moves = [
                (child.winning_frac(game_state.next_player), child.move, child.num_rollouts)
                for child in root.children
            ]
            scored_moves.sort(key=lambda x: x[0], reverse=True)
            for s, m, n in scored_moves[:10]:
                print('%s - %.3f (%d)' % (m, s, n))

# tag::mcts-selection[]
        # Having performed as many MCTS rounds as we have time for, we
        # now pick a move.
        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_frac(game_state.next_player)
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        if best_pct < 0.1:
            return Move(is_resign=True)
        if self.output:
            print('Select move %s with win pct %.3f' % (best_move, best_pct))
        return best_move
# end::mcts-selection[]

    """
    This function is no longer random, but returns the child that has 
    the highest uct score 
    """
# tag::mcts-uct[]
    def select_child(self, node):
        """Select a child according to the upper confidence bound for
        trees (UCT) metric.
        """
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = math.log(total_rollouts)

        best_score = -1
        best_child = None
        # Loop over each child.
        for child in node.children:
            # Calculate the UCT score.
            win_percentage = child.winning_frac(node.game_state.next_player)
            exploration_factor = math.sqrt(log_rollouts / child.num_rollouts)
            uct_score = win_percentage + self.temperature * exploration_factor
            # Check if this is the largest we've seen so far.
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child
# end::mcts-uct[]

    @staticmethod
    def simulate_random_game(game):
        bots = {
            Player.black: FastRandomBot(),
            Player.white: FastRandomBot(),
        }
        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
        return game.winner()