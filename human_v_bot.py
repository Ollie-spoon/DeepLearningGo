from dlgo.agent.naive import RandomAgent
from dlgo.mcts.mcts import MCTSAgent
from dlgo import goboard_fast as goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move, human_move_selection
from six.moves import input


def main():
    board_size = 9
    game = goboard.GameState.new_game(board_size)
    ## agent = RandomAgent()
    player_colour = gotypes.Player.black
    agent = MCTSAgent(10000, 1.7)

    while not game.is_over():
        print(chr(27) + "[2J")
        print_board(game.board)
        if game.next_player == player_colour:
            human_move = input('-- ')
            move = human_move_selection(human_move, goboard)
        else:
            move = agent.select_move(game)
        print_move(game.next_player, move)
        game = game.apply_move(move)
    print("The " + str(game.winner())[7:] + " player wins")


if __name__ == '__main__':
    main()