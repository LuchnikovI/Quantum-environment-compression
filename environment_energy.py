import jax.numpy as jnp
from functools import reduce
from jax.scipy.linalg import expm
from jax import jit


def state_from_embedding(emb_state, isometries, spin_number):
    """ Returns the decoded system-environment state from the embedding
        Args:
            isometries: isometric matrices from truncation procedure
            last_embedding_state: resulting embedding vector
        Returns: system-environment statevector
    """
    def convolution(state, isometry):
        state = state.reshape(isometry.shape[1], -1)
        return jnp.tensordot(isometry, state, axes=((1), (0)))

    initial_state = emb_state.reshape(-1, 2)
    decoded_vec = reduce(convolution, isometries[::-1], initial_state).reshape(-1)
    return decoded_vec.reshape(spin_number * [2]).T.reshape(-1)


def energy_from_embedding(emb_state, isometries, coluplings, fields):



    return energy
