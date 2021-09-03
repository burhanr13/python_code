def bin_packing(items, size):
    items = sorted(items, reverse=True)
    space = [size]
    res = [[]]
    not_fit = []
    for item in items:
        if item > size:
            not_fit += [item]
            continue
        for j in range(len(space)):
            if space[j] >= item:
                space[j] -= item
                res[j] += [item]
                break
        else:
            space += [size-item]
            res += [[item]]

    print(f"Result for array {items} and bin size {size}:")
    for i in res:
        print(i)
    print(f"Total bins used: {len(res)}")
    print(f"Total used space: {sum(items)-sum(not_fit)}")
    print(f"Total empty space: {sum(space)}")
    if len(not_fit) > 0:
        print(f"Couldn't fit: {not_fit}")

    return res, space


bin_packing([15.38]*8+[19.38]*4+[22.5]*4+[30.38]*4+[54.38]*2+[72]*14, 96)
