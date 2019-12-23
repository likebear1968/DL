import numpy as np

def scale(count, typ=0):
    if typ == 0:
        return np.sqrt(2.0 / count) # He
    return np.sqrt(1.0 / count)     # Xavier

def id_to_category(obs, prd):
    y = np.reshape(prd, (-1, np.shape(prd)[-1]))
    if np.size(y) == np.size(obs):
        t = np.reshape(obs, np.shape(y))
    else:
        if not isinstance(obs, (list, np.ndarray)):
            obs = [obs]
        t = np.zeros_like(y, dtype=int)
        t[np.arange(np.shape(y)[0]), np.array(obs)] = 1
    return np.reshape(t, np.shape(prd))

def category_to_id(obs, categories, offset=0):
    t = np.reshape([0] * np.size(obs), np.shape(obs))
    for i, v in enumerate(categories):
        t[np.where(obs == v)] = i + offset
    return t
        