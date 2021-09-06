import jax.numpy as jnp
from functools import reduce
from jax.scipy.linalg import expm


pauli = jnp.array([jnp.array([[1., 0.], [0., 1.]], jnp.complex64),
                jnp.array([[0., 1.], [1., 0.]], jnp.complex64),
                jnp.array([[0., -1j], [1j, 0.]], jnp.complex64),
                jnp.array([[1., 0.], [0., -1.]], jnp.complex64)])


def decode_embedding(input_isometries, last_embedding_state):
    """ Return decoded system-environment state from embedding
        Args:
            isometries: isometric matrices from truncation procedure
            last_embedding_state: resulting embedding vector
        Returns: system-environment statevector
    """

    isometries = input_isometries.conj()
    reshaped_isometries = [isometries[0]]
    for i in range(len(isometries) - 1):
        if isometries[i].shape[1] != isometries[i + 1].shape[0]:
            add_spins = int( jnp.log(isometries[2].shape[0] /
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


def exact_unitary(couplings, fields, time_interval=None):
    """ Return exact hamiltonian and evolution operator
        ==============================================================
        | Test function |
        Do not use for large dimentions to prevent RAM overconsumption
        ==============================================================
    """
    def operator_product(oper1, oper2):
        return jnp.kron(oper1, oper2)

    n = fields.shape[0]
    hamiltonian = sum([sum([reduce(lambda x, y: jnp.kron(x, y), oper_list)
                       for oper_list in [couplings[i, k] * jnp.array(
                       i * [pauli[0]] +  2 * [pauli[k + 1]] + \
                       (n - i - 2) * [pauli[0]]) for i in range(n - 1)]])
                       for k in range(3)]) + \
                  sum([sum([reduce(lambda x, y: jnp.kron(x, y), oper_list)
                       for oper_list in [fields[i, k] * jnp.array(
                       i * [pauli[0]] + [pauli[k + 1]] + \
                       (n - i - 1) * [pauli[0]]) for i in range(n)]])
                   for k in range(3)])

    if time_interval != None:
        return expm(-1j * time_interval * hamiltonian)
    else:
        return hamiltonian


def exact_dynamics(hamiltonian, initial_state, tau, time_steps):
    """ Calculate exact dynamics of the whole system
        ==============================================================
        | Test function |
        Do not use for large dimentions to prevent RAM overconsumption
        ==============================================================
    """
    times = jnp.arange(time_steps) * tau
    conseq_states = []
    for time in times:
        conseq_states.append(expm(-1j * time * hamiltonian) @ initial_state)
    return conseq_states

