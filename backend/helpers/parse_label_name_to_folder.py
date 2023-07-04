def parse_label_name(label):
    """add alphabetic variable to label name for tensorflow data set creation"""
    if label == 'scratch':
        return 'a_scratch'
    if label == 'dent':
        return 'b_dent'
    if label == 'rim':
        return 'c_rim'
    return 'd_other'
