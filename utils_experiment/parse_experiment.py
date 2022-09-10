import experiments


def parse_experiment(function):
    def wrapper(*args, **kwargs):

        experiment = getattr(experiments, kwargs["experiment"])

        return function(*args, **experiment)

    return wrapper
