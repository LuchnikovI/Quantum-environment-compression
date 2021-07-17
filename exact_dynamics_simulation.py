import jax.numpy as jnp
from jax import lax, jit, vmap
from functools import partial


@vmap
def _mut_inf(rho):
    h12 = -jnp.log((rho * rho.T).sum())
    rho = rho.reshape((2, 2, 2, 2))
    rho1 = jnp.einsum('iqjq->ij', rho)
    rho2 = jnp.einsum('qiqj->ij', rho)
    h1 = -jnp.log((rho1 * rho1.T).sum())
    h2 = -jnp.log((rho2 * rho2.T).sum())
    return h1 + h2 - h12


def _one_qubit_desn(rhos):
    rho1 = jnp.einsum('iqjq->ij', rhos[0].reshape((2, 2, 2, 2)))[jnp.newaxis]
    rho2 = jnp.einsum('pqiqj->pij', rhos.reshape((-1, 2, 2, 2, 2)))
    return jnp.concatenate([rho1, rho2], axis=0)


@partial(jit, static_argnums=(2, 3, 5))
def dynamics(gates,
             state,
             depth,
             n,
             control_seq=None,
             use_control=False):
    """This function allows simulating dynamics of a spin chain consisting
    of an odd number of spins.

    Args:
        gates: complex valued array of shape (n-1, 4, 4),
            two qubit unitary gates
        state: complex valued array of shape (n, 2),
            initial state of each spin, overall state is separable
        depth: int value representing depth of a circuit
        n: int value representing number of spins
        control_seq: None, of complex valued array of shape (depth, 2, 2),
            unitary gates representing a control sequance
        use_control: boolean value showing whether one uses control or not

    Returns:
        rhos: complex valued array of shape (depth, n, 2, 2), density matrices
            of each spin at each discrete time moment
        mut_inf: real valued array of shape (depth, n-1), mutual information
            between the first spin and each other spin"""

    def iter_over_in_state(total_state, spin_state):
        return jnp.tensordot(state.reshape((2, -1)).sum(0), spin_state, axes=0), None
    in_state = jnp.concatenate([jnp.array([1.]), jnp.zeros((2 ** n - 1,))], axis=0)
    state, _ = lax.scan(iter_over_in_state, in_state, state)
        
    first_layer = gates[::2]
    second_layer = gates[1::2]
    def iter_over_gates(state, gate):
        state = state.reshape((4, -1))
        state = jnp.tensordot(state, gate, [[0], [1]])
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
        mut_inf = _mut_inf(rhos)
        rhos = _one_qubit_desn(rhos)
        state, _ = lax.scan(iter_over_gates, state, first_layer)
        state = state.reshape((2, -1))
        state = state.T
        state = state.reshape((2, -1))
        state = state.T
        state = state.reshape((-1,))
        state, _ = lax.scan(iter_over_gates, state, second_layer)
        if use_control:
            state = state.reshape((2, -1))
            state = jnp.tensordot(control, state, axes=1)
            state = state.reshape((-1,))
        return state, (rhos, mut_inf)
    _, (rhos, mut_inf) = lax.scan(iter_over_layers, state, xs=control_seq, length=depth)
    return rhos, mut_inf.real
