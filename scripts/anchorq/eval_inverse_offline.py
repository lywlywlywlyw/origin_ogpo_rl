"""Pure forward/reverse audit on paired offline states; no environments or training."""
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from ogpo.agents.anchorq import AnchorQAgent
from ogpo.agents.dsrl import DSRLBestOfNAgent
from ogpo.networks import ActorVectorField
from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    result = dict(count=x.size, nonfinite=int(x.size-finite.size))
    if finite.size:
        result.update(mean=float(finite.mean()), **dict(zip(
            ['p50', 'p90', 'p95', 'p99', 'max'],
            map(float, np.percentile(finite, [50, 90, 95, 99, 100])))))
    result.update({f'fraction_gt_{t}': float(np.mean(x > t)) for t in [10, 100, 1000]})
    return result


def main():
    run = Path('exp/ogpo/OGPO/square/square-mh-low_dim/anchorq_square_shared_ogpo_bc_v2seed1')
    output = Path('run_logs/anchorq_inverse_offline')
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run/'flags.json').read_text())
    cfg = config['agent']
    cfg['noise_dim'] = cfg['horizon_length'] * 7
    checkpoint = config['checkpoint']['restore_bc_flow_path']
    with open(checkpoint, 'rb') as f:
        params = pickle.load(f)['agent']['actor_network']['params']['modules_actor']
    assert cfg['encoder'] is None and cfg['time_embedding'] == 'sinusoidal'
    model = ActorVectorField(hidden_dims=tuple(cfg['actor_hidden_dims']), action_dim=28,
        layer_norm=cfg['actor_layer_norm'],
        time_embedding=SinusoidalTimeEmbedding(embed_dim=cfg['time_embedding_dim']))

    class Decoder:
        def select(self, name):
            assert name == 'actor_bc_flow'
            return lambda s, x, t, **kw: model.apply({'params': params}, s, x, t, **kw)

    dummy = SimpleNamespace(config=cfg, bc_flow_network=Decoder())
    forward = jax.jit(lambda s, z: DSRLBestOfNAgent.compute_flow_actions.__wrapped__(dummy, s, noises=z))
    reverse = jax.jit(lambda s, a: AnchorQAgent.reverse_flow_actions.__wrapped__(dummy, s, a))
    refs = np.load(run/'anchorq_references.npz')['reference_actions']
    assert refs.shape == (6, 28)
    dataset = '/root/datasets/square/mh/low_dim_v15.hdf5'
    rows, demos, timesteps = [], [], []
    with h5py.File(dataset, 'r') as f:
        for name in sorted(f['data'], key=lambda x: int(x[5:])):
            obs = np.concatenate([f[f'data/{name}/obs/{k}'][:] for k in
                ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object']], axis=-1)
            rows.append(obs.astype(np.float32))
            demos.extend([int(name[5:])] * len(obs))
            timesteps.extend(range(len(obs)))
    all_obs = np.concatenate(rows)
    seed = 20261023
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(all_obs), 4096, replace=False)
    observations = all_obs[ids]
    np.savez(output/'states.npz', observations=observations, dataset_indices=ids,
        demo_ids=np.asarray(demos)[ids], timesteps=np.asarray(timesteps)[ids], reference_actions=refs)
    metrics = {kind: {k: [] for k in ['z_rms', 'action_l2', 'boundary_fraction', 'xyz_l2', 'rotation_l2', 'gripper_l2', 'decoded_gripper_abs']}
               for kind in ['bc', 'reference']}
    for start in range(0, len(observations), 128):
        s = jnp.asarray(observations[start:start+128])
        z0 = jnp.asarray(rng.normal(size=(len(s), 28)).astype(np.float32))
        bc = forward(s, z0)
        for kind in metrics:
            obs = s if kind == 'bc' else jnp.repeat(s, 6, axis=0)
            a = bc if kind == 'bc' else jnp.tile(jnp.asarray(refs), (len(s), 1))
            z = reverse(obs, a)
            reconstructed = forward(obs, z)
            delta = (reconstructed-a).reshape(-1, 4, 7)
            for k, value in dict(xyz_l2=jnp.linalg.norm(delta[:, :, :3].reshape(-1, 12), axis=-1),
                rotation_l2=jnp.linalg.norm(delta[:, :, 3:6].reshape(-1, 12), axis=-1),
                gripper_l2=jnp.linalg.norm(delta[:, :, 6], axis=-1),
                decoded_gripper_abs=jnp.mean(jnp.abs(reconstructed.reshape(-1, 4, 7)[:, :, 6]), axis=-1),
                z_rms=jnp.sqrt(jnp.mean(z*z, axis=-1)),
                action_l2=jnp.linalg.norm(reconstructed-a, axis=-1),
                boundary_fraction=jnp.mean(jnp.abs(a) >= .999999, axis=-1)).items():
                metrics[kind][k].append(np.asarray(value))
        if start % 512 == 0:
            print(f'{start+128}/{len(observations)} states', flush=True)
    summary = dict(checkpoint=checkpoint, dataset=dataset, dataset_rows=len(all_obs),
        seed=seed, states=len(observations), flow_steps=cfg['flow_steps'],
        forward_final_clip=True, reverse_clip=False, online_evaluated=False, results={})
    for kind, fields in metrics.items():
        arrays = {k: np.concatenate(v) for k, v in fields.items()}
        np.savez(output/f'{kind}.npz', **arrays, dataset_indices=ids)
        summary['results'][kind] = {k: stats(v) for k, v in arrays.items()}
        if kind == 'reference':
            summary['results'][kind]['directions'] = {
                name: {k: stats(v.reshape(-1, 6)[:, i]) for k, v in arrays.items()}
                for i, name in enumerate(['+x', '-x', '+y', '-y', '+z', '-z'])}
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
