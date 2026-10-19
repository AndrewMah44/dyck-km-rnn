import jax.numpy as jnp

def plot_probs_by_condition(ax, probs, condition, 
                            color='k', alpha=0.25, linestyle='-',
                            include_null=False):

    if include_null:
        x = range(probs.shape[2])
        y = jnp.mean(probs[condition], axis=0)
        yerr = jnp.std(probs[condition], 0)

    else:
        x = range(probs.shape[2]-1)
        y = jnp.mean(probs[condition], axis=0)[:-1]
        yerr = jnp.std(probs[condition], 0)[:-1]

    ax.plot(x, y, color=color, linestyle=linestyle)
    ax.fill_between(x, y+yerr, y-yerr, color=color, alpha=alpha)

    return ax
