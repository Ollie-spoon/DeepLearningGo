import importlib


class Encoder:
    """
    The encoder base class has the following methods:
        name - returns the name of the encoder
        encode - encodes a gamestate
        encode_point - turns a go board point into an integer index
        decode_point_index - turns an index back into a point
        num_points - 
        shape - shape of the encoded board structure
    """
    def name(self):
        raise NotImplementedError()

    def encode(self, game_state):
        raise NotImplementedError()
    
    def encode_point(self, point):
        raise NotImplementedError()
    
    def decode_point_index(self, index):
        raise NotImplementedError()
    
    def num_points(self):
        raise NotImplementedError()
    
    def shape(self):
        raise NotImplementedError()


"""
get_encoder_by_name 
    if board size is one integer then make board dimensions of size board_size
    use the name of the encoder to create a constructor of size board_size
    each encoder module will have a 'create' function to return an instance of it
"""
def get_encoder_by_name(name, board_size):
    
    if isinstance(board_size, int):
        board_size = (board_size, board_size)

    module = importlib.import_module('dlgo.encoders.' + name)
    constructor = getattr(module, 'create')

    return constructor(board_size)