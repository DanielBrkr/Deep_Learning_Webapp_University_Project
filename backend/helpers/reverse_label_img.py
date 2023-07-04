def reverse_label_img(label):
    """re-translating labels"""
    if label == 0:
        return 'scratch'
    if label == 1:
        return 'dent'
    if label == 2:
        return 'rim'
    return 'other'
