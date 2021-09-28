import jax.numpy as jnp
from functools import reduce
from jax.scipy.linalg import expm
from jax import jit



def decode_embedding(embedding_state, isometries):
    """ Returns the decoded system-environment state from the embedding
        Args:
            isometries: isometric matrices from truncation procedure
            last_embedding_state: resulting embedding vector
        Returns: system-environment statevector
    """
    def convolution(state, iso):
        state = state.reshape(-1, iso.shape[1])
        return jnp.tensordot(state, iso, axes=((-1), (-1)))

    in_state = embedding_state.reshape(-1, 2).T
    return reduce(convolution, isometries[::-1], in_state).reshape(-1)
