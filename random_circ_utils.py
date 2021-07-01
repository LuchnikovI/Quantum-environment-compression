from jax import numpy as jnp
from jax import random


def _one_gate_gen(subkey):
    """Generates two-qubit quantum gate"""

    u = random.normal(subkey, (4, 4, 2))
    u = u[..., 0] + 1j * u[..., 1]
    u, _ = jnp.linalg.qr(u)
    return u


def random_circuit_generation(key,
                              depth,
                              num_of_qubits):
    """Generates checkerboard quantum circuit"""

    circ =  [[_one_gate_gen(subsubkey) for subsubkey in random.split(subkey, depth)] for subkey in random.split(key, num_of_qubits)]
    return circ


def _blocks_conv(blocks, mode):
    down_block, up_block = blocks
    if mode == 'fwd':
        mpo_block = jnp.tensordot(down_block, up_block, axes=1).transpose((0, 2, 1, 3))
    else:
        mpo_block = jnp.tensordot(up_block, down_block, axes=1)
    return mpo_block


def _split_gate(gate):
    gate = gate.reshape((2, 2, 2, 2))
    gate = gate.transpose((0, 2, 1, 3))
    gate = gate.reshape((4, 4))
    v, s, wh = jnp.linalg.svd(gate)
    s_sqrt = jnp.sqrt(s)
    v = v * s_sqrt
    wh = wh * s_sqrt[:, jnp.newaxis]
    v = v.reshape((2, 2, 4))
    v = v.transpose((0, 2, 1))
    wh = wh.reshape((4, 2, 2))
    wh = wh.transpose((1, 0, 2))
    return wh, v


def checkerboard2square(circ):
    """Converts a checkerboard circuit into the corresponding square circuit"""
    up_blocks = []
    down_blocks = []
    mpo_blocks = []
    for i, line in enumerate(circ):
        if i == 0:
            down_mps, up_block = zip(*map(_split_gate, line))
            up_blocks.append(up_block)
        elif i == len(circ)-1:
            down_block, up_mps = zip(*map(_split_gate, line))
            down_blocks.append(down_block)
        else:
            down_block, up_block = zip(*map(_split_gate, line))
            down_blocks.append(down_block)
            up_blocks.append(up_block)
    for i, (down_block, up_block) in enumerate(zip(down_blocks, up_blocks)):
        mode = 'fwd' if i % 2 == 0 else 'rev'
        mpo_block = list(map(lambda x: _blocks_conv(x, mode=mode), zip(down_block, up_block)))
        last_mpo = mpo_block[0][..., 0][..., jnp.newaxis]
        mpo_block = mpo_block[:0:-1] + [last_mpo]
        mpo_blocks.append(mpo_block)
    down_mps = list(down_mps[::-1])
    last_up_mps = up_mps[0][..., 0][..., jnp.newaxis]
    up_mps = list(up_mps[:0:-1]) + [last_up_mps]
    return [down_mps] + mpo_blocks + [up_mps]
