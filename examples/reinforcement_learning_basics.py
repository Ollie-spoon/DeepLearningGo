"""
Reinforcement learning summary

Unfortunately for go we don't know which moves are good and which ones
are bad, we only know which games we have won and which we haven't. We
want to be able to calcuate the return of each individual move. This is
actually easier than it might seem at first.

We can create an action return function   <1>, to produce the return for each
move.

There are a number of useful numpy functions that can allow us to select
moves from our distribution of generated moves probabilistically. The first
of these is   <2>   which allows us to select a value from a list based on a
probability distribution either with or without removal. The second is   <3>
where we clip the probabilities. This ensures that no moves are overreacted
to. If the computer assigns a very high score to a move that isn't that good,
we still want to allow other moves to be randomly selected or it will never
learn that this is a bad move.

The key, it seems, to reinforcement learning is not individual games with
individually good moves, but in having many many games where the same ideas
can be tested out over and over again in different circumstances and to
learn from this.

For this example, we then create a policy agent  <4>. The purpose of the policy
agent is to select moves.
"""

import numpy as np
import h5py
from keras import Sequential
from keras.layers import Dense, Activation
## from tensorflow.keras.models import save_model

from dlgo import kerasutil
from dlgo.agent.base import Agent
from dlgo.encoders import simple
from dlgo.encoders.base import get_encoder_by_name
from dlgo.networks import large
from examples.save_test import save_model, load_model_from_checkpoint


# <1>
# reward[exp_idx] is the reward that the agent saw immediately after this action
# Note that this is psuedocode
# also note that you can add a discounted return to this by multiplying by a factor each time,
# example [100%, 75%, 56%, ...]

def action_return(exp_length):
    total_return = np.zeros(exp_length)
    for exp_idx in range(exp_length):
        total_return[exp_idx] = reward[exp_idx]  # noqa
        for future_reward_idx in range(exp_idx + 1, exp_length):
            total_return[exp_idx] += reward[future_reward_idx]  # noqa
    return total_return


# <2>

def rps(replace):
    return np.random.choice(
        ['rock', 'paper', 'scissors'],
        p=[0.5, 0.3, 0.2],
        size=3,
        replace=replace
    )


# <3>

def clip_probs(original_probs):
    min_p = 1e-5
    max_p = 1 - min_p
    clipped_probs = np.clip(original_probs, min_p, max_p)
    clipped_probs = clipped_probs / np.sum(clipped_probs)
    return clipped_probs

# <4>


class PolicyAgent(Agent):
    def __init__(self, model, encoder, network_name):
        super(PolicyAgent, self).__init__()
        self.model = model
        self.encoder = encoder
        self.network_name = network_name

    def save(self, training_steps: str = None):
        save_model(self.model, network_name=self.network_name, encoder_name=self.encoder.name(), training_steps=training_steps)
        print("PolicyAgent 'save': Successful.")


def load_policy_agent(file_name):
    model, network_name, encoder_name, output_dim = load_model_from_checkpoint(file_name)

    board_size = np.sqrt(output_dim)
    assert board_size == np.round(board_size)

    encoder = get_encoder_by_name(encoder_name, (board_size, board_size))
    agent = PolicyAgent(model, encoder, network_name)
    print("PolicyAgent 'load': Successful.")
    return agent


if __name__ == '__main__':
    print("rps(): " + str(rps(True)))
    board_size = 19
    encoder = simple.SimpleEncoder((board_size, board_size))
    model = Sequential()
    for layer in large.layers(encoder.shape()):
        model.add(layer)
    model.add(Dense(encoder.num_points()))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    agent = PolicyAgent(model, encoder, 'large')
    agent.save()


    # agent = load_policy_agent('large_simple.h5')
    # print("agent: " + str(agent))
    # agent.model.summary()
    # print("agent.encoder: " + str(agent.encoder.name()))
