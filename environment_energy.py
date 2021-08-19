import jax.numpy as jnp
from functools import reduce
from jax.scipy.linalg import expm


pauli = jnp.array([jnp.array([[1., 0.], [0., 1.]], jnp.complex64),
                jnp.array([[0., 1.], [1., 0.]], jnp.complex64),
                jnp.array([[0., -1j], [1j, 0.]], jnp.complex64),
                jnp.array([[1., 0.], [0., -1.]], jnp.complex64)])


def decode_embedding(isometries, last_embedding_state):
    """ Return decoded system-environment state from embedding
        Args:
            isometries: isometric matrices from truncation procedure
            last_embedding_state: resulting embedding vector

        Returns: system-environment statevector
    """
    reshaped_isometries = [isometries[0]]
    for i in range(len(isometries) - 1):
        if isometries[i + 1].shape[0] != isometries[i].shape[1]:
            reshaped_isometries.append(isometries[i + 1].reshape(
                                -1, 2, isometries[i + 1].shape[1]))
        else:
            reshaped_isometries.append(isometries[i + 1])

    def stack_isometries(iso1, iso2):
        return jnp.tensordot(iso1, iso2, axes=((-1), (0)))

    iso_stack = reduce(stack_isometries, reshaped_isometries)
    return jnp.tensordot(last_embedding_state.reshape(-1, 2),
                         iso_stack, axes=((0), (-1))).reshape(-1)



def calculate_exact_unitary(couplins, fields, time_interval=None):
    """ Return exact hamiltonian and evolution operator
        ==============================================================
        | Test function |
        Do not use for large dimentions to prevent RAM overconsumption
        ==============================================================
    """
    def operator_product(oper1, oper2):
        return jnp.kron(oper1, oper2)

    n = fields.shape[0]
    hamiltonian = sum([sum([reduce(operator_product,
                    jnp.array(i * [pauli[0]] + 2 * [pauli[k]] + \
                    (n - i - 2) * [pauli[0]])) for i in range(
                     n - 1)]) for k in range(3)]) + \
                  sum([sum([reduce(operator_product,
                    jnp.array(i * [pauli[0]] + [pauli[k]] + \
                    (n - i - 1) * [pauli[0]])) for i in range(
                     n - 1)]) for k in range(3)])
    if time_interval != None:
        return expm(-1j * time_interval * hamiltonian)
    else:
        return hamiltonian




