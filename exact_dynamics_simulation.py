import jax.numpy as jnp
from jax import lax, jit, vmap
from functools import partial
from jax.scipy.special import xlogy


@vmap
def _mut_inf(rho):
    lmbd12 = jnp.linalg.eigvalsh(rho)
    lmbd12 = jnp.maximum(lmbd12, 0)
    h12 = -xlogy(lmbd12, lmbd12).sum()
    rho = rho.reshape((2, 2, 2, 2))
    rho1 = jnp.einsum('iqjq->ij', rho)
    rho2 = jnp.einsum('qiqj->ij', rho)
    lmbd1 = jnp.linalg.eigvalsh(rho1)
    lmbd1 = jnp.maximum(lmbd1, 0)
    h1 = -xlogy(lmbd1, lmbd1).sum()
    lmbd2 = jnp.linalg.eigvalsh(rho2)
    lmbd2 = jnp.maximum(lmbd2, 0)
    h2 = -xlogy(lmbd2, lmbd2).sum()
    return h1 + h2 - h12


def _one_qubit_dens(rhos):
    rho1 = jnp.einsum('piqjq->pij', rhos[0].reshape((1, 2, 2, 2, 2)))
    rho2 = jnp.einsum('pqiqj->pij', rhos.reshape((-1, 2, 2, 2, 2)))
    return jnp.concatenate([rho1, rho2], axis=0)


@partial(jit, static_argnums=(2, 3, 5, 6))
def choi(gates,
         state,
         depth,
         n,
         control_seq=None,
         use_control=False,
         dtype=jnp.complex64):
    """This function returns Choi matrices describing quantum channels with
    input in 0-th position and output in any position for all time steps.

    Args:
        gates: complex valued array of shape (n-1, 4, 4),
            two qubit unitary gates
        state: complex valued array of shape (n-1, 2),
            initial state of each spin, overall state is separable
        depth: int value representing depth of a circuit
        n: int value representing number of spins
        control_seq: None, of complex valued array of shape (depth, 2, 2),
            unitary gates representing a control sequance
        use_control: boolean value showing whether one uses control or not

    Returns:
        rhos: complex valued array of shape (depth, n, 4, 4), choi matrices"""

    def iter_over_in_state(total_state, spin_state):
        return jnp.tensordot(total_state.reshape((2, -1))[0], spin_state, axes=0).reshape((-1,)), None
    in_state = jnp.concatenate([jnp.array([1.], dtype=dtype), jnp.zeros((2 ** n - 1,), dtype=dtype)], axis=0)
    state, _ = lax.scan(iter_over_in_state, in_state, state)
    state = jnp.tensordot(jnp.eye(4, dtype=dtype) / 2, state, axes=0)
    state = state.reshape((-1,))
        
    first_layer = gates[::2]
    second_layer = gates[1::2]
    def iter_over_gates(state, gate):
        state = state.reshape((2, 4, -1))
        state = jnp.tensordot(state, gate, [[1], [1]])
        return state.reshape((-1,)), None
    def iter_over_qubits(state, x):
        state = state.reshape((4, -1))
        rho = (state[:, jnp.newaxis] * state[jnp.newaxis].conj()).sum(-1)
        state = state.reshape((2, 2, -1))
        state = state.transpose((0, 2, 1))
        state = state.reshape((-1,))
        return state, rho
    def iter_over_layers(state, control):
        _, rhos = lax.scan(iter_over_qubits, state, xs=None, length=n-1)
        state, _ = lax.scan(iter_over_gates, state, first_layer)
        state = state.reshape((2, 4, -1))
        state = state.transpose((0, 2, 1))
        state = state.reshape((-1,))
        state, _ = lax.scan(iter_over_gates, state, second_layer)
        if use_control:
            state = state.reshape((2, 2, -1))
            state = jnp.tensordot(control, state, [[1], [1]])
            state = state.transpose((1, 0, 2))
            state = state.reshape((-1,))
        return state, rhos
    _, rhos = lax.scan(iter_over_layers, state, xs=control_seq, length=depth)
    return rhos

def dynamics(in_state, choi):
    circ_shape = choi.shape[:2]
    choi = choi.reshape((*circ_shape, 2, 2, 2, 2))
    rhos = jnp.einsum('qpkimj,k,m->qpij', choi, in_state, in_state.conj())
    return rhos
