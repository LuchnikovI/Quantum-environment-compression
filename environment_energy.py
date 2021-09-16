import jax.numpy as jnp
from functools import reduce
from jax.scipy.linalg import expm
from jax import jit


def decode_embedding(input_isometries, last_embedding_state):
    """ Return decoded system-environment state from embedding
        Args:
            isometries: isometric matrices from truncation procedure
            last_embedding_state: resulting embedding vector
        Returns: system-environment statevector
    """

    isometries = list(map(lambda x: jnp.conj(x), input_isometries))
    reshaped_isometries = [isometries[0]]
    for i in range(len(isometries) - 1):
        if isometries[i].shape[1] != isometries[i + 1].shape[0]:
            add_spins = int(jnp.log(isometries[2].shape[0] /
                                    isometries[1].shape[1]
                                    ) / jnp.log(2))

            reshaped_isometries.append(isometries[i + 1].reshape(
                                        -1, jnp.power(2, add_spins),
                                        isometries[i + 1].shape[1]))
        else:
            reshaped_isometries.append(isometries[i + 1])

    def stack_isometries(iso1, iso2):
        return jnp.tensordot(iso1, iso2, axes=((-1), (0)))

    iso_stack = reduce(stack_isometries, reshaped_isometries)
    return jnp.tensordot(last_embedding_state.reshape(-1, 2),
                         iso_stack, axes=((0), (-1))).reshape(-1)

