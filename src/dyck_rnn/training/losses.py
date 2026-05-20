import jax
from optax import softmax_cross_entropy_with_integer_labels

def pred_loss_func(model, x, y, mask):
    logits = model(x)
    loss = softmax_cross_entropy_with_integer_labels(logits, y) * mask

    return sum(loss) / sum(mask)