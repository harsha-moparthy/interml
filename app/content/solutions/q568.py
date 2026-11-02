def q568(training_gradients, public_validation_gradient, learning_rate):
    """Return first-order predicted validation LOSS REDUCTION for one example step: lr *
    dot(g_train,g_val). Inputs are flattened [N,P] and [P]. Positive values predict
    improvement. Validation is assumed public; this function is not itself a private
    selector or a submodularity proof.
    """
    return learning_rate * (training_gradients @ public_validation_gradient)
