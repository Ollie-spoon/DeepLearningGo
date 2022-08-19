from dlgo.agent.naive_fast import FastRandomBot as RandomAgent
from dlgo.mcts.mcts import MCTSAgent
from dlgo import goboard_fast as goboard
## This ^^^ can be changed to goboard_slow/goboard/goboard_fast
# for a 13x13 board:
# goboard_slow: 0.1816 s/move
# goboard:      0.1873 s/move
# goboard_fast: 0.002383 s/move ~ 75x faster than the other two

from dlgo import gotypes
from dlgo.utils import print_board, print_move
import time


def main():
    board_size = 5
    output = False
    bots = {
        gotypes.Player.black: MCTSAgent(100, 1.75, False),
        gotypes.Player.white: MCTSAgent(100, 1.75, False),
    }
    wins = [0, 0]
    if output:
        no_of_moves = 0
        start = time.time()
    for i in range(0, 40):
        game = goboard.GameState.new_game(board_size)
        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
            if output:
                print(chr(27) + "[2J")
                print_board(game.board)
                print_move(game.next_player, bot_move)
                no_of_moves +=1
        if output:
            print("The " + str(game.winner())[7:] + " player wins")
            end = time.time()-start
            print("Time taken to do " + str(no_of_moves) + " moves : " + str(end))
            print("Average of %f s per move." % (end/no_of_moves))
        winner = 0 if str(game.winner())[7:] == "black" else 1
        wins[winner] += 1
        print("Game " + str(i+1) + " done. Winner: " + str(game.winner())[7:])
    print("wins: " + str(wins))


if __name__ == '__main__':
    main()

