import jax.random as jr
from nanogpt.data.batching import batch_iterator

def compute_validation_loss(model, dataset, batch_size, loss_fn):
    num_batches = dataset.shape[0] / batch_size
    key = jr.PRNGKey(1)

    loss = 0.0
    for x, y in batch_iterator(dataset, batch_size, key=key):
        loss += loss_fn(model, x, y, key)

    return loss / num_batches
