import jax.numpy as jnp
from environment_processing import Environment


def _to_mpo(gates):
    gates = gates.reshape((-1, 2, 2, 2, 2))
    gates = gates.transpose((0, 1, 3, 2, 4))
    gates = gates.reshape((-1, 4, 4))
    u, s, vh = jnp.linalg.svd(gates)
    up_block = u * jnp.sqrt(s)[:, jnp.newaxis]
    up_block = up_block.reshape((-1, 2, 2, 4))
    up_block = up_block.transpose((0, 1, 3, 2))
    down_block = vh * jnp.sqrt(s)[..., jnp.newaxis]
    down_block = down_block.reshape((-1, 4, 2, 2))
    down_block = down_block.transpose((0, 2, 1, 3))
    system_block = up_block[0]
    last_block = down_block[-1]
    up_block = up_block[1:]
    down_block = down_block[:-1]
    def combine(i, up_block, down_block):
        if i % 2 == 0:
            mpo_block = jnp.tensordot(up_block[i], down_block[i], axes=1)
            mpo_block = mpo_block.transpose((0, 2, 1, 3))
        else:
            mpo_block = jnp.tensordot(down_block[i], up_block[i], axes=1)
        return mpo_block
    return [system_block] + [combine(i, up_block, down_block) for i in range(down_block.shape[0])] + [last_block]


def embedding(gates,
              in_state,
              depth,
              max_dim,
              eps):
    """Returns effective model predictiing dynamics of the 0-th spin.

    Args:
        gates: complex valued array of shape (n-1, 4, 4),
            two qubit unitary gates
        in_state: complex valued array of shape (n, 2),
            initial state of each spin, overall state is separable
        depth: int value representing depth of a circuit
        max_dim: int value, if environment dim reaches this value,
            then it is being truncated
        eps: float value, admissible truncation error"""

    in_state = [x for x in in_state]
    mpo = _to_mpo(gates)
    system_block = depth * [mpo[0]]
    mpo = mpo[1:]
    mpo_in = [jnp.tensordot(mpo_block, state[:, jnp.newaxis], axes=1) for mpo_block, state in zip(mpo, in_state)]
    mpo = (depth - 1) * [mpo] + [mpo_in]
    mpo = list(zip(*mpo))
    environment = Environment()
    env = mpo[-1]
    for mpo_block in mpo[-2::-1]:
        env = environment.add_subsystem(mpo_block, env)
        if env[0].shape[0] > max_dim:
            env, log_norm = environment.set_to_canonical(env)
            norm, env = environment.truncate_canonical(env, eps)
            print(norm)
            if env[0].shape[0] > max_dim:
                print('dim = {}'.format(env[0].shape[0]))
    env, log_norm = environment.set_to_canonical(env)
    norm, env = environment.truncate_canonical(env, eps)
    print(norm)
    return environment.build_system(system_block, env)


def wire_embedding(gates,
                   in_state,
                   depth,
                   max_dim,
                   eps):
    """Returns effective model predicting dynamics of the 0-th and last spins.

    Args:
        gates: complex valued array of shape (n-1, 4, 4),
            two qubit unitary gates
        in_state: complex valued array of shape (n, 2),
            initial state of each spin, overall state is separable
        depth: int value representing depth of a circuit
        max_dim: int value, if environment dim reaches this value,
            then it is being truncated
        eps: float value, admissible truncation error"""

    def _mpo2mps(ker):
        shape = ker.shape
        return ker.reshape((shape[0], 16, shape[-1]))
    def _mps2mpo(ker):
        shape = ker.shape
        return ker.reshape((shape[0], 4, 4, shape[-1]))

    in_state = [x for x in in_state]
    mpo = _to_mpo(gates)
    system_block1 = depth * [mpo[0]]
    system_block2 = depth * [mpo[-1]]
    mpo = mpo[1:-1]
    mpo_in = [jnp.tensordot(mpo_block, state[:, jnp.newaxis], axes=1) for mpo_block, state in zip(mpo, in_state)]
    mpo = (depth - 1) * [mpo] + [mpo_in]
    mpo = list(zip(*mpo))
    environment = Environment()
    env = mpo[-2]
    for mpo_block in mpo[-3::-1]:
        env = environment.combine_subsystems(mpo_block, env)
        if env[0].shape[0] > max_dim:
            env = [_mpo2mps(ker) for ker in env]
            env, log_norm = environment.set_to_canonical(env)
            norm, env = environment.truncate_canonical(env, eps)
            print(norm)
            if env[0].shape[0] > max_dim:
                print('dim = {}'.format(env[0].shape[0]))
    env = [_mps2mpo(ker) for ker in env]
    env = environment.add_subsystem(env, system_block2)
    return environment.build_system(system_block1, env)


def dynamics_with_embedding(embedding_matrices,
                            in_state,
                            use_control=False,
                            control_seq=None):
    """Returns dynamics of a system from embedding matrices.

    Args:
        embedding_matrices: list of complex valued matrices
        in_state: complex valued array of shape (2,)
        use_control: boolean flag showing whether to use control
            or not
        control_seq: None, or complex valued array of shape (depth, 2, 2),
            unitary gates representing a control sequance

    Returns:
        complex valued array of shape (time_steps, 2, 2)"""

    sys_rhos = []

    for i, transition_matrix in enumerate(embedding_matrices[::-1]):
        sys_rho = in_state.reshape((-1, 2))
        sys_rho = sys_rho[..., jnp.newaxis] * sys_rho[:, jnp.newaxis].conj()
        sys_rho = sys_rho.sum(0)
        sys_rhos.append(sys_rho)
        in_state = jnp.tensordot(transition_matrix, in_state, axes=1)
        in_state = in_state / jnp.linalg.norm(in_state)
        if use_control:
            in_state = in_state.reshape((-1, 2))
            in_state = jnp.tensordot(in_state, control_seq[i], axes=[[1], [1]])
            in_state = in_state.reshape((-1,))

    return jnp.array(sys_rhos)

def dynamics_with_wire_embedding(embedding_matrices,
                                 in_state):
    """Returns dynamics of a systems from embedding matrices.

    Args:
        embedding_matrices: list of complex valued matrices
        in_state: complex valued array of shape (2,)
        use_control: boolean flag showing whether to use control
            or not
        control_seq: None, or complex valued array of shape (depth, 2, 2),
            unitary gates representing a control sequance

    Returns:
        two complex valued arrays of shape (time_steps, 2, 2)"""

    sys1_rhos = []
    sys2_rhos = []

    for i, transition_matrix in enumerate(embedding_matrices[::-1]):
        sys_rho = in_state.reshape((-1, 2))
        sys_rho = sys_rho[..., jnp.newaxis] * sys_rho[:, jnp.newaxis].conj()
        sys_rho = sys_rho.sum(0)
        sys1_rhos.append(sys_rho)
        sys_rho = in_state.reshape((2, -1))
        sys_rho = sys_rho.T
        sys_rho = sys_rho[..., jnp.newaxis] * sys_rho[:, jnp.newaxis].conj()
        sys_rho = sys_rho.sum(0)
        sys2_rhos.append(sys_rho)
        in_state = jnp.tensordot(transition_matrix, in_state, axes=1)
        in_state = in_state / jnp.linalg.norm(in_state)

    return jnp.array(sys1_rhos), jnp.array(sys2_rhos)
