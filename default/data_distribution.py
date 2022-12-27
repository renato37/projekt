from sklearn.model_selection import train_test_split


def splitData(X, y, perc):
    return train_test_split(X, y, test_size=perc, random_state=0)
