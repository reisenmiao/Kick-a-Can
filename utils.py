def soft_update(target, source, learning_rate):

    for target_param, param in zip(target.parameters(), source.parameters()):

        target_param.data.copy_(
            target_param * (1.0 - learning_rate) + param.data * learning_rate
        )