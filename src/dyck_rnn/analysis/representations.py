import jax.numpy as jnp
from sklearn.decomposition import PCA

def get_centroids(hidden_activity, states, n_states, 
                  cond = None, return_pca=False):
    if cond is not None:
        hidden_activity = hidden_activity[cond]
        states = states[cond]
            
    counts = jnp.bincount(states, length=n_states)
    sums = jnp.zeros(
        (n_states, hidden_activity.shape[1]), 
        dtype=hidden_activity.dtype).at[states].add(hidden_activity)

    centroids = jnp.where(
        counts[:, None] > 0,
        sums / counts[:, None],
        jnp.nan,
    )

    if return_pca:
        pca = PCA(hidden_activity.shape[1]).fit(centroids)
        return centroids, pca

    return centroids

def get_centroid_drift(hidden_activity, states, n_states, early, late):
    early_centroids = get_centroids(
        hidden_activity, 
        states,
        n_states,
        early)

    late_centroids = get_centroids(
        hidden_activity, 
        states,
        n_states,
        late)

    delta = late_centroids - early_centroids
    return early_centroids, late_centroids, delta