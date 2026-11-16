c = [97, 98, 97, 98, 99]
to_merge = (97, 98)


tokens = []
i = 0

while i < len(c):
    if i + 1 < len(c) and (c[i], c[i + 1]) == to_merge:
        tokens.append(256)
        i += 1
    else:
        tokens.append(c[i])

    i += 1


print(tokens)
