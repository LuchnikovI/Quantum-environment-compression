from jax import numpy as jnp, jit
from jax import random
from functools import reduce, partial
from jax.scipy.linalg import expm
from tqdm import tqdm


class  Floquet_dynamics:
    """ Floquet dynamics calculation for
        transverce field Ising model

        Input:

        Output:
    """

    def __init__(self, couplings, local_fields, env_size, tau, number_of_steps):
        #Pauli matrices
        self.identity = jnp.array([[1., 0.], [0., 1.]],
                                        jnp.complex64)
        self.pauli = jnp.array([jnp.array([[0., 1.], [1., 0.]],
                                                jnp.complex64),
                                jnp.array([[0., -1j], [1j, 0.]],
                                                jnp.complex64),
                                jnp.array([[1., 0.], [0., -1.]],
                                                jnp.complex64)])
        self.pauli_prods = jnp.array([jnp.kron(oper, oper)
                             for oper in self.pauli])
        self.id_pauli = jnp.array([jnp.kron(self.identity, oper)
                             for oper in self.pauli])
        self.pauli_id = jnp.array([jnp.kron(oper, self.identity)
                             for oper in self.pauli])

        lmbd, u = jnp.linalg.eigh(self.pauli + self.identity)
        u = u.transpose((0, 2, 1)).conj()
        self.sq_sigma = jnp.sqrt(lmbd)[..., jnp.newaxis] * u

        #Hamiltonian parameters
        self.couplings = couplings
        #required shape = (number_site_connections, 3)
        self.local_fields= local_fields
        #required shape = (number_of_sites, 3)

        #Simulation parameters
        self.env_size = env_size
        self.number_of_sites = env_size + 1
        self.tau = tau
        self.number_of_steps = number_of_steps

        #State initialization
        self.initial_state = self.init_state()

        #Ganerate evolution MPO
        self.layer = self.construct_checkerboard()


    def init_state(self, sys_state=None, env_state=None):
        """ Initialize spin chain environment in all-up state """

        if sys_state == None:
            sys_state = jnp.array([1., 0.], jnp.complex64)
        else:
            pass
        if env_state == None:
            env_state = jnp.eye(1, 2**self.env_size, dtype=jnp.complex64)
        else:
            pass
        return jnp.tensordot(sys_state, #system state
                    env_state, #environment state
                    axes=0).reshape((self.env_size + 1) * (2, ))


    def construct_checkerboard(self):
        """ Construct checkerboard gate layer from
            hamiltonian parameters.
        """
        mul = jnp.array([2.] + (self.local_fields.shape[0] - 2) * [1.] + [2.])
        # correct fields to checkerboard construction
        fields_corr = jnp.multiply(mul, self.local_fields.T).T
        checker_ham = jnp.tensordot(self.couplings, self.pauli_prods,
                                    axes=((1), (0))) + \
                      jnp.tensordot(fields_corr[:-1, :] / 2, self.pauli_id,
                                    axes=((1), (0))) + \
                    jnp.tensordot(fields_corr[1:, :] / 2, self.id_pauli,
                                    axes=((1), (0)))
        # matrix exponential 
        gate_layer = jnp.array(list(map(lambda operator: expm(
                     -1j * self.tau * operator), checker_ham)))
        return gate_layer.reshape(self.env_size, 2, 2, 2, 2)


    @staticmethod
    def _apply_gate(state, argument, size):
        evol_oper, state_axes = argument
        assert state_axes[0] < state_axes[1], 'Incorrect state axes to contract'
        updated_state = jnp.tensordot(state, operator, axes=(state_axes, (2, 3)))
        trans_order = list(range(state_axes[0])) +\
                      [size - 2] +\
                      list(range(states_axes[0], state_axes[1] - 1)) +\
                      [size - 1] +\
                      list(range(state_axes[1]-1, size-2))
        return jnp.transpose(updated_state, trans_order)

    @staticmethod
    def _apply_layer(state, layer, sites_number):
        """ Move state forward in time for one step """

        def _apply_gate(state, argument, size):
            """ Apply gate """
            evol_oper, state_axes = argument
            #assert state_axes[0] < state_axes[1], 'Incorrect state axes to contract'
            updated_state = jnp.tensordot(state, evol_oper,
                                          axes=(state_axes, (2, 3)))
            trans_order = list(range(state_axes[0])) +\
                          [size - 2] +\
                          list(range(state_axes[0], state_axes[1] - 1)) +\
                          [size - 1] +\
                          list(range(state_axes[1]-1, size-2))
            return jnp.transpose(updated_state, trans_order)

        inds = jnp.arange(0, sites_number - 1)
        even_inds = [(ind, ind + 1) for ind in inds[::2]]
        odd_inds = [(ind, ind + 1) for ind in inds[1::2]]
        even_layer = layer[::2]
        odd_layer = layer[1::2]
        even_state = reduce(lambda st, lr: _apply_gate(st, lr, sites_number),
                                          zip(even_layer, even_inds), state)
        odd_state = reduce(lambda st, lr: _apply_gate(st, lr, sites_number),
                                      zip(odd_layer, odd_inds), even_state)
        return odd_state


    def _bloch_projection(self, state, state_axes):
        state = jnp.tensordot(state, self.sq_sigma,
                              axes=[[state_axes], [2]])
        state = state.reshape((-1, 3, 2))
        return (jnp.abs(state) ** 2).sum((0, 2)) - 1


    def state_evolution(self):
        assert self.initial_state != None, 'Initial state is None'
        assert self.layer != None, 'Operator layer is None '

        in_state = self.initial_state
        bloch_vectors = [[self._bloch_projection(
                         in_state, state_ax) for state_ax in range(
                                            self.number_of_sites)]]
        for i in tqdm(range(self.number_of_steps), desc="Exact dynamics simulation", ncols=100):
            in_state= self._apply_layer(in_state, self.layer, self.number_of_sites)
            bloch_vectors.append([self._bloch_projection(
                in_state, state_ax) for state_ax in range(self.number_of_sites)])
        return jnp.array(bloch_vectors)

