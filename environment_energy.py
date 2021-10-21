import jax.numpy as jnp
from functools import reduce
from jax.scipy.linalg import expm
from jax import jit


identity = jnp.array([[1., 0.], [0., 1.]], jnp.complex64)
paulix = jnp.array([[0., 1.], [1., 0.]], jnp.complex64)
pauliy = jnp.array([[0., -1j], [1j, 0.]], jnp.complex64)
pauliz = jnp.array([[1., 0.], [0., -1.]], jnp.complex64

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


def hamiltonian_to_mpo(couplings, fields):
    """ Local hamiltonian in MPO form """
    local_parts = jnp.tensordot(fields, pauli, axes=((1), (0)))
    left_vec = jnp.array([local_parts[0]] + list((couplings[0] *
                                        pauli.T).T) + [identity])
    right_vec = jnp.array([identity] + list(pauli) + [local_parts[-1]])

    zeros = 4 * [jnp.zeros((2, 2), dtype=jnp.complex64)]
    const_rows = jnp.array([[identity] + zeros, [pauli[0]] + zeros,
                             [pauli[1]] + zeros, [pauli[2]] + zeros])
    mid_blocks = []
    for i in range(1, len(fields) - 1):
        site_row = jnp.array([local_parts[i]] + list((couplings[i] *
                                                pauli.T).T) + [identity])
        block = jnp.concatenate([const_rows, site_row[jnp.newaxis, :]], axis=0)
        mid_blocks.append(block)
    return [left_vec] + mid_blocks + [right_vec]



def energy_calc(emb_state, isometries, mpo):
    """ Calculate energy of embedded state """

    @jit
    def mpo_contract(stack, oper):
        """ Contraction with MPO """
        stack_d = stack.shape[-1]
        stack = jnp.tensordot(stack, oper.T, axes=((0), (-1)))
        stack = stack.transpose((4, 0, 3, 1, 2))
        stack = stack.reshape((stack.shape[0],) + 2 * (stack_d * 2,))
        return stack

    def net_contract(state, isometry, mpo):
        process_tensor, i = state # unpacking 
        iso_dim, proc_shape = isometry.shape[0], process_tensor.shape
        mpo_num = int(jnp.log2(iso_dim / proc_shape[-1]))
        # number of additional spins
        process_tensor = reduce(mpo_contract, mpo[i:i + mpo_num],
                                process_tensor) # contract mpo
        process_tensor = jnp.tensordot(process_tensor, isometry,
                                                axes=((1), (0))) # right isometry 
        process_tensor = jnp.tensordot(process_tensor, isometry.conj(),
                                                axes=((1), (0))) # left isometry 
        i += mpo_num
        return (process_tensor, i)


    energy = reduce(lambda state, iso: net_contract(state, iso, mpo[1:-1]),
                                                 isometries, (mpo[0], 0))
    emb_state = emb_state.reshape(-1, 2)
    energy = jnp.tensordot(energy[0], emb_state, axes=((1), (0)))
    energy = jnp.tensordot(energy, emb_state.conj(), axes=((1), (0)))
    return jnp.tensordot(energy, mpo[-1], axes=((0, 1, 2), (0, 1, 2)))

