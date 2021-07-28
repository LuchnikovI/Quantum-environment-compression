from jax import numpy as jnp
from jax import random
from functools import reduce

def _push_dens(ker,
               u,
               lmbd,
               eps=1e-5):
    """Helper function for pushing dens. matrix forward in time."""

    # pushing dens. matrix
    ker = jnp.tensordot(ker, u, axes=1)
    dim, rank = ker.shape[1:]
    ker = ker.reshape((-1, dim*rank))
    q, r = jnp.linalg.qr(ker)
    r = r.reshape((-1, dim, rank))
    r = r * lmbd
    r = r.reshape((-1, dim*rank))
    u, lmbd, _ = jnp.linalg.svd(r, full_matrices=False)

    # setting threshold
    sq_norm = (lmbd ** 2).sum()
    cum_sq_norm = jnp.cumsum((lmbd ** 2)[::-1])
    trshld = (jnp.sqrt(cum_sq_norm / sq_norm) > eps).sum()

    # truncation
    u = u[:, :trshld]
    lmbd = lmbd[:trshld]
    u = q @ u
    ker = jnp.tensordot(u.conj().T, ker.reshape((-1, dim, rank)), axes=1)
    return u, lmbd, ker


def _push_r(ker, r):
    """Helper function for pushing orth. center backward in time."""

    _, dim, right_bond = ker.shape
    ker = jnp.tensordot(r, ker, axes=1)
    ker = ker.reshape((-1, right_bond))
    ker, r = jnp.linalg.qr(ker)
    ker = ker.reshape((-1, dim, right_bond))
    return ker, r


def _push_dens_naive(ker,
                     rho):
    """Helper function for pushing dens. matrix forward in time (naive)."""

    rho = jnp.einsum('iqk,kn,jqn->ij',
                     ker,
                     rho,
                     ker.conj(),
                     optimize='optimal')
    return rho


class Environment:

    def __init__(self):
        pass

    def init_random(self,
                    key,
                    n,
                    dim,
                    chi):

        def sample(shape, key):
            ker = random.normal(key, shape + (2,))
            ker = ker[..., 0] + 1j * ker[..., 1]
            return ker
        keys = random.split(key, n)
        mps = [sample((chi, dim, chi), key) for key in keys[:-2]]
        mps = [sample((chi, dim, chi), keys[-2])] + mps + [sample((chi, dim, 1), keys[-1])]
        return mps

    def norm(self,
             state):
        """Caluclates norm of a MPS directly.

        Args:
            state: list of arrays, mps kernels

        Return:
            value of log(norm)"""

        def iter(vars, ker):
            log_norm, rho = vars
            rho = _push_dens_naive(ker, rho)
            norm = jnp.linalg.norm(rho)
            log_norm = log_norm + jnp.log(norm)
            rho = rho / norm
            return log_norm, rho
        list_to_reduce = [(jnp.array(0.), jnp.array([[1.]]))] + state[::-1]
        log_norm, final_rho = reduce(iter, list_to_reduce)
        final_norm = jnp.trace(final_rho)
        log_norm = log_norm + jnp.log(final_norm)
        return log_norm.real / 2

    def set_to_canonical(self,
                         state):
        """Set environment to the canonical form.

        Args:
            state: list of arrays, mps representation of an
                environment

        Returns:
            state: list of arrays, canonical form of the environment mps
            log_norm: scalar, value of log norm"""

        def iter_canonic(vars, ker):
            updated_state, r, log_norm = vars
            ker, r = _push_r(ker, r)
            norm = jnp.linalg.norm(r)
            r = r / norm
            log_norm = log_norm + jnp.log(norm)
            updated_state.append(ker)
            return updated_state, r, log_norm

        list_to_reduce = [([], jnp.eye(state[0].shape[0]), jnp.array(0.))] + state
        state, _, log_norm = reduce(iter_canonic, list_to_reduce)
        return state, log_norm

    def truncate_canonical(self,
                           state,
                           eps=1e-5):
        """Truncate an environment in the canonical form.

        Args:
            state: list of arrays, mps kernels
            eps: singular values cut off threshold

        Returns:
            norm: norm of truncated canonical mps
            state: truncated state"""

        def iter_trunc(vars, ker):
            updated_state, u, lmbd = vars
            u, lmbd, ker = _push_dens(ker, u, lmbd, eps)
            updated_state.append(ker)
            return updated_state, u, lmbd
        list_to_reduce = [([], jnp.array([[1.]]), jnp.array([1.]))] + state[::-1]
        state, final_u, final_lmbd = reduce(iter_trunc, list_to_reduce)
        return jnp.sqrt((final_lmbd ** 2).sum()), state[::-1]

    def add_subsystem(self,
                      subsystem_ker,
                      env_ker):
        """Add subsystem to the environment.

        Args:
            subsystem_ker: list of array like of shape
                (sys_dim, int_rank, int_rank, sys_dim),
                representing mpo kernel of the subsystem
            env_ker: list of array like of shape (env_dim, int_rank, env_dim)
                representing mps kernel of the environment

        Returns:
            list of array like of shape (env_dim * sys_dim, int_rank, env_dim * sys_dim),
                representing new mps kernel of the environment"""

        def f(kernels):
            subsystem_ker, env_ker = kernels
            new_right_bond = subsystem_ker.shape[-1] * env_ker.shape[-1]
            env_ker = jnp.tensordot(subsystem_ker, env_ker, [[2], [1]])
            env_ker = env_ker.transpose((3, 0, 1, 4, 2))
            env_ker = env_ker.reshape((-1, 4, new_right_bond))
            return env_ker
        return list(map(f, zip(subsystem_ker, env_ker)))

    def combine_subsystems(self,
                           subsystem_ker1,
                           subsystem_ker2):
        """Combine two subsystems.

        Args:
            subsystem_ker1: list of array like of shape
                (sys_dim1, int_rank, int_rank, sys_dim1),
                representing mpo kernel of the subsystem
            subsystem_ker2: list of array like of shape
                (sys_dim2, int_rank, int_rank, sys_dim2),
                representing mpo kernel of the subsystem

        Returns:
            list of array like of shape (sys_dim1 * sys_dim2, int_rank, int_rank, sys_dim1 * sys_dim2)"""

        def f(kernels):
            ker1, ker2 = kernels
            new_right_bond = ker1.shape[-1] * ker2.shape[-1]
            env_ker = jnp.tensordot(ker1, ker2, [[2], [1]])
            env_ker = env_ker.transpose((3, 0, 1, 4, 5, 2))
            env_ker = env_ker.reshape((-1, 4, 4, new_right_bond))
            return env_ker
        return list(map(f, zip(subsystem_ker1, subsystem_ker2)))

    def build_system(self,
                     system_ker,
                     env_ker):
        """Combines the system and its environment.

        Args:
            system_ker: list of array like of shape (sys_dim, int_rank, sys_dim)
                representing mps kernel of the system
            env_ker: list of array like of shape (env_dim, int_rank, env_dim)
                representing mps kernel of the environment

        Returns:
            list of array like of shape (env_dim * sys_dim, env_dim * sys_dim)
            representing transition matrices"""

        def f(kernels):
            system_ker, env_ker = kernels
            new_right_bond = system_ker.shape[-1] * env_ker.shape[-1]
            sys_ker = jnp.tensordot(system_ker, env_ker, [[1], [1]])
            sys_ker = sys_ker.transpose((2, 0, 3, 1))
            sys_ker = sys_ker.reshape((-1, new_right_bond))
            return sys_ker
        return list(map(f, zip(system_ker, env_ker)))

    def dynamics(self,
                 transition_matrices,
                 in_state,
                 use_control=False,
                 cntrl_seq=None):
        """Calculates dynamics of the subsystem.

        Args:
            transition_matrices: list of array like of shape
                (env_dim * sys_dim, env_dim * sys_dim)
            in_state: array like of shape (sys_dim,)
            use_control: boolean flag showing if to use control seq. or not
            cntrl_seq: None or list of array like of shape (2, 2),
                control seq.

        Returns:
            list of array like of shape (sys_dim, sys_dim),
            density matrices of the system"""

        sys_rhos = []
        sys_rho = in_state.reshape((-1, 2))
        sys_rho = sys_rho[..., jnp.newaxis] * sys_rho[:, jnp.newaxis].conj()
        sys_rho = sys_rho.sum(0)
        sys_rhos.append(sys_rho)

        for i, transition_matrix in enumerate(transition_matrices[::-1]):
            in_state = jnp.tensordot(transition_matrix, in_state, axes=1)
            in_state = in_state / jnp.linalg.norm(in_state)
            if use_control:
                in_state = in_state.reshape((-1, 2))
                in_state = jnp.tensordot(in_state, cntrl_seq[i], axes=[[1], [1]])
                in_state = in_state.reshape((-1,))
            sys_rho = in_state.reshape((-1, 2))
            sys_rho = sys_rho[..., jnp.newaxis] * sys_rho[:, jnp.newaxis].conj()
            sys_rho = sys_rho.sum(0)
            sys_rhos.append(sys_rho)

        return jnp.array(sys_rhos)
