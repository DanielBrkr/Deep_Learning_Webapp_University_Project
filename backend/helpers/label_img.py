def label_img(label):
    """defining labels"""
    if label == 'a_scratch':
        return 0
    if label == 'b_dent':
        return 1
    if label == 'c_rim':
        return 2
    return 3
