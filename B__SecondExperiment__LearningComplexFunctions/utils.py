def limit_iterations(iterator, limit=1_000):
    for _, point in zip(range(limit, iterator)):
        yield point
