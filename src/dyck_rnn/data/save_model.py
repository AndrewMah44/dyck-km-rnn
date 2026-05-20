import equinox as eqx

def save_model(model, filename):

    # split model into params (arrays) and static (non-arrays / functions)
    params, _ = eqx.partition(model, eqx.is_inexact_array)

    # serialise and save learned params
    eqx.tree_serialise_leaves(filename, params)
