"""
This is being done on pytorch as conda does not support tf_nightly at the
moment which is the package containing the 'MultiHeadAttention' layer.

Basics on Transformers:
Transformers work on the idea of relational attention. If you look at an
        image of a bird, a small section around the wing will likely have
        a strong connection with another small section of the birds beak.
        The conclusion here is that these two sections have high
        attention due to their relationship. Equally in a sentence,

                'The blue bird hurt it's wing,'

        hurt has a high attention with both bird and with wing, but a low
        attention with blue.

        The basic idea is that if you can give a word, or section of an
        image a vector, you can find the dot product of the vectors of two
        different words/sections to find the attention of the two. Vectors
        that are highly aligned will be close to 1, and vectors that are
        not will be close to 0.

        Attention follows the mathematical formula,

                Attention( Q, K, V ) = softmax(QK^(T)/√dk) * V,

        where Q is a matrix of queries and K is a matrix of keys, each
        query and key with dimension d_k, and V is  a matrix of values.

               "The key/value/query concept is analogous to retrieval systems. For
                example, when you search for videos on Youtube, the search engine
                will map your query (text in the search bar) against a set of keys
                (video title, description, etc.) associated with candidate videos
                in their database, then present you the best matched videos (values)."

        Self-Attention refers to that fact that in a lot of use cases,
        queries, keys, and values all come from the same source.

Data Preprocessing:
        For an image, it is typical to split the image up into a number of
        discrete sections as there is a limit to the computational power
        available.

        Step 1:  Reshaping each 'patch' into a 1D vector separately.
        Various functions can be applied to do this, but one easy one is a
        Conv2d with a 'Stride'== patch width.

        Step 2:  Position Embedding.
        This can either be done with a Sinosoidal function (original paper)
        or far more commonly learnable embedding or rotary embedding.

Multi-Head:
        Literally just multiple parallel self attention layers side by side
        with the same architecture allows for an increase in nuance of
        learned attention than just one head.

        The Multi head attention block is typically made from a normalisation layer used
"""