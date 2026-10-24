"""Online action coverage against strictly earlier physical transitions.

Nearest neighbours use fixed, offline-standardized state coordinates. Actions
use their executed, normalized coordinates (one physical action, never latent
noise or a flattened action chunk). All history is retained by default.
"""
import numpy as np
from scipy.spatial import cKDTree


class ActionCoverage:
    def __init__(self, state_scale, action_dim, *, k=32, epsilon=.1,
                 sample_interval=1, rebuild_interval=5000):
        self.scale = np.asarray(state_scale, dtype=np.float64).reshape(-1)
        if not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError('Coverage state scales must be finite and positive')
        for name, value in [('k', k), ('sample_interval', sample_interval),
                            ('rebuild_interval', rebuild_interval)]:
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f'coverage {name} must be a positive integer')
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('coverage epsilon must be positive')
        self.k, self.epsilon = int(k), float(epsilon)
        self.sample_interval, self.rebuild_interval = int(sample_interval), int(rebuild_interval)
        self.states = np.empty((4096, len(self.scale)), dtype=np.float32)
        self.actions = np.empty((4096, action_dim), dtype=np.float32)
        self.size = self.queries = self.duplicates = 0
        self.tree = None
        self.tree_size = 0
        self.action_min = np.full(action_dim, np.inf)
        self.action_max = np.full(action_dim, -np.inf)
        self._window = []

    @classmethod
    def from_dataset(cls, dataset, config, action_dim):
        if not config.get('cal_action_coverage', False):
            return None
        # Fixed scales make distances comparable across time and algorithms.
        field = 'full_states' if 'full_states' in dataset._dict else 'observations'
        scale = np.maximum(np.asarray(dataset[field]).std(axis=0), 1e-3)
        return cls(scale, action_dim, k=config.get('coverage_k', 32),
                   epsilon=config.get('coverage_epsilon_a', .1),
                   sample_interval=config.get('coverage_sample_interval', 1),
                   rebuild_interval=config.get('coverage_rebuild_interval', 5000))

    def _neighbours(self, state):
        if self.tree is None or self.size - self.tree_size >= self.rebuild_interval:
            self.tree_size = self.size
            # Separate snapshot: the append buffers grow, but the tree stays valid.
            self.tree = cKDTree(self.states[:self.size], copy_data=True)
        _, indices = self.tree.query(state, k=min(self.k, self.tree_size))
        candidates = np.concatenate([np.atleast_1d(indices).astype(int),
                                     np.arange(self.tree_size, self.size)])
        distances = np.linalg.norm(self.states[candidates] - state, axis=1)
        # Merge the tree's exact KNN with every transition since its rebuild.
        nearest = np.argsort(distances, kind='stable')[:self.k]
        return candidates[nearest], distances[nearest]

    def observe(self, observation, action):
        state = np.asarray(observation, dtype=np.float64).reshape(-1) / self.scale
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if state.shape != self.scale.shape or action.shape != self.action_min.shape:
            raise ValueError('Action coverage state/action dimensions do not match the dataset')
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError('Non-finite action coverage input')
        result = None
        if self.size >= self.k and self.size % self.sample_interval == 0:
            indices, distances = self._neighbours(state)
            local = self.actions[indices].astype(np.float64)
            novelty = float(np.linalg.norm(local - action, axis=1).min())
            covariance = np.atleast_2d(np.cov(local, rowvar=False)) if self.k > 1 else np.zeros((len(action), len(action)))
            span = np.ptp(local, axis=0)
            eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.)
            result = dict(novelty=novelty, duplicate=float(novelty < self.epsilon),
                          state_knn_radius=float(distances[-1]),
                          local_action_span_mean=float(span.mean()),
                          local_action_span_l2=float(np.linalg.norm(span)),
                          local_cov_trace=float(np.trace(covariance)),
                          local_cov_max_eigenvalue=float(eigenvalues[-1]))
            for i in range(len(action)):
                result[f'local_action_span/{i}'] = float(span[i])
                for j in range(i, len(action)):
                    result[f'local_cov/{i}_{j}'] = float(covariance[i, j])
            self.queries += 1
            self.duplicates += int(novelty < self.epsilon)
            self._window.append(result)
        # Insert after querying: never compare a transition to itself.
        if self.size == len(self.states):
            self.states = np.concatenate([self.states, np.empty_like(self.states)])
            self.actions = np.concatenate([self.actions, np.empty_like(self.actions)])
        self.states[self.size] = state
        self.actions[self.size] = action
        self.size += 1
        self.action_min = np.minimum(self.action_min, action)
        self.action_max = np.maximum(self.action_max, action)
        return result

    def metrics(self):
        result = dict(history_size=self.size, query_count=self.queries,
                      duplicate_rate=self.duplicates / self.queries if self.queries else 0.,
                      window_query_count=len(self._window))
        keys = ['novelty', 'duplicate', 'state_knn_radius', 'local_action_span_mean',
                'local_action_span_l2', 'local_cov_trace', 'local_cov_max_eigenvalue']
        for i in range(len(self.action_min)):
            keys.append(f'local_action_span/{i}')
            keys.extend(f'local_cov/{i}_{j}' for j in range(i, len(self.action_min)))
        # Stable columns even when the first log precedes the first KNN query.
        result.update({key + '_mean': float('nan') for key in keys})
        result.update(novelty_p10=float('nan'), novelty_p50=float('nan'),
                      novelty_p90=float('nan'), window_duplicate_rate=float('nan'))
        if self._window:
            for key in self._window[0]:
                result[key + '_mean'] = float(np.mean([row[key] for row in self._window]))
            values = [row['novelty'] for row in self._window]
            result.update(novelty_p10=float(np.quantile(values, .1)),
                          novelty_p50=float(np.quantile(values, .5)),
                          novelty_p90=float(np.quantile(values, .9)),
                          window_duplicate_rate=float(np.mean([r['duplicate'] for r in self._window])))
            self._window.clear()
        if self.size:
            for i, (low, high) in enumerate(zip(self.action_min, self.action_max)):
                result[f'action_min/{i}'] = float(low)
                result[f'action_max/{i}'] = float(high)
                result[f'action_span/{i}'] = float(high - low)
            result['action_span_l2'] = float(np.linalg.norm(self.action_max - self.action_min))
        return result
