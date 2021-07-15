from jax import numpy as jnp
from functools import reduce


sigma = jnp.array([[[0, 1], [1, 0]],
                   [[0, -1j], [1j, 0]],
                   [[1, 0], [0, -1]]], jnp.complex64)
sigma_psd = sigma + jnp.eye(2)
lmbd, u = jnp.linalg.eigh(sigma_psd)
u = u.transpose((0, 2, 1)).conj()
sq_sigma = jnp.sqrt(lmbd)[..., jnp.newaxis] * u
Id = jnp.eye(2)


def _apply_gate(state,
                sides,
                gate,
                n):
    """Helper function for the exact dynamics simulation"""

    n = len(state.shape)
    gate = gate.reshape((2, 2, 2, 2))
    if sides[0] > sides[1]:
        gate = gate.transpose((1, 0, 3, 2))
        sides = sides[::-1]
    state = jnp.tensordot(state, gate, [sides, [2, 3]])
    new_order = list(range(sides[0])) + [n-2] +  list(range(sides[0], sides[1]-1)) + [n-1] + list(range(sides[1]-1, n-2))
    state = state.transpose(new_order)
    return state


def _apply_sigma(state,
                 side):
    """Helper function for the exact dynamics simulation"""

    state = jnp.tensordot(state, sq_sigma, axes=[[side], [2]])
    state = state.reshape((-1, 3, 2))
    return (jnp.abs(state) ** 2).sum((0, 2)) - 1


def _partial_density(state,
                     sides):
    """Helper function for the exact dynamics simulation"""

    min_idx = min(sides)
    max_idx = max(sides)
    state = jnp.tensordot(state, Id, axes=[[max_idx], [1]])
    state = jnp.tensordot(state, Id, axes=[[min_idx], [1]])
    state = state.reshape((-1, 4))
    state = state[..., jnp.newaxis] * state[:, jnp.newaxis].conj()
    state = state.sum(0)
    return state

def _mutual_inf(state):
    """Helper function for the exact dynamics simulation"""

    eps = 1e-6
    state = state.reshape(2, 2, 2, 2)
    rho1 = jnp.trace(state, axis1=1, axis2=3)
    rho2 = jnp.trace(state, axis1=0, axis2=2)
    whole_spec = jnp.linalg.eigvalsh(state)
    spec1 = jnp.linalg.eigvalsh(rho1)
    spec2 = jnp.linalg.eigvalsh(rho2)
    return -(spec1 * jnp.log(spec1 + eps)).sum() - (spec2 * jnp.log(spec2 + eps)).sum() + (whole_spec * jnp.log(whole_spec + eps)).sum()


class ExactFloquet:

    def __init__(self,
                 n):
        """Initializes an object of the class ExactFloquet.

        Args:
            n: int, number of qubits."""

        self.n = n

    def simulate(self,
                 in_state,
                 layer,
                 time_steps,
                 use_control=False,
                 cntrl_seq=None,
                 mutual_inf=False):
        """Simulate dynamics of a qubit system.

        Args:
            in_state: array like of shape self.n * (2,)
            layer: list of arrays of shape (4, 4)
            time_steps: int, number of time steps
            use_control: boolean flag showing whether to use control seq. or not
            cntrl_seq: None or list of array like of shape (2, 2),
                control seq
            mutual_inf: boolean flag showing whether to calculate mutual inf.
                or not

        Returns:
            bloch_vecs: array like of shape (time_steps, self.n, 3),
                array of Bloch vectors
            mutual_inf: array like of shape (time_steps, self.n)
                mutual information"""

        first_layer = layer[::2]
        second_layer = layer[1::2]
        first_layer_sides = [(2*i+1, 2*i) for i in range(len(first_layer))]
        second_layer_sides = [(2*i+2, 2*i+1) for i in range(len(second_layer))]
        apply_layer = lambda state, gate_sides: _apply_gate(state, gate_sides[0], gate_sides[1], self.n)
        in_state = in_state.reshape(self.n * (2,))
        if mutual_inf:
            inf = [_mutual_inf(_partial_density(in_state, [side, self.n-1])) for side in range(self.n-1)]
            inf_layers = [inf]
        rho_layer = [_apply_sigma(in_state, side) for side in range(self.n)]
        rho_layers = [rho_layer]
        for i in range(time_steps):
            to_reduce = [in_state] + list(zip(first_layer_sides, first_layer)) + list(zip(second_layer_sides, second_layer))
            in_state = reduce(apply_layer, to_reduce)
            if use_control:
                in_state = jnp.tensordot(cntrl_seq[i], in_state, axes=1)
            rho_layer = [_apply_sigma(in_state, side) for side in range(self.n)]
            rho_layers.append(rho_layer)
            if mutual_inf:
                inf = [_mutual_inf(_partial_density(in_state, [side, self.n-1])) for side in range(self.n-1)]
                inf_layers.append(inf)
        if mutual_inf:
            return jnp.array(rho_layers), jnp.array(inf_layers)
        else:
            return jnp.array(rho_layers)
