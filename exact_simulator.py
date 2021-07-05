from jax import numpy as jnp
from functools import reduce


sigma = jnp.array([[[0, 1], [1, 0]],
                   [[0, -1j], [1j, 0]],
                   [[1, 0], [0, -1]]], jnp.complex64)
sigma_psd = sigma + jnp.eye(2)
lmbd, u = jnp.linalg.eigh(sigma_psd)
u = u.transpose((0, 2, 1)).conj()
sq_sigma = jnp.sqrt(lmbd)[..., jnp.newaxis] * u


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
                 cntrl_seq=None):
        """Simulate dynamics of a qubit system.

        Args:
            in_state: array like of shape self.n * (2,)
            layer: list of arrays of shape (4, 4)
            time_steps: int, number of time steps
            use_control: boolean flag showing if to use control seq. or not
            cntrl_seq: None or list of array like of shape (2, 2),
                control seq.

        Returns:
            array like of shape (time_steps, self.n, 3),
            array of Bloch vectors"""

        first_layer = layer[::2]
        second_layer = layer[1::2]
        first_layer_sides = [(2*i+1, 2*i) for i in range(len(first_layer))]
        second_layer_sides = [(2*i+2, 2*i+1) for i in range(len(second_layer))]
        apply_layer = lambda state, gate_sides: _apply_gate(state, gate_sides[0], gate_sides[1], self.n)
        rho_layers = []
        in_state = in_state.reshape(self.n * (2,))
        rho_layer = [_apply_sigma(in_state, side) for side in range(self.n)]
        rho_layers.append(rho_layer)
        for i in range(time_steps):
            to_reduce = [in_state] + list(zip(first_layer_sides, first_layer)) + list(zip(second_layer_sides, second_layer))
            in_state = reduce(apply_layer, to_reduce)
            if use_control:
                in_state = jnp.tensordot(cntrl_seq[i], in_state, axes=1)
            rho_layer = [_apply_sigma(in_state, side) for side in range(self.n)]
            rho_layers.append(rho_layer)
        return jnp.array(rho_layers)