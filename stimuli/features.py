def zcr(data):
    zerocrossing = 0
    for i in range(1, len(data)):
        if (data[i - 1]) > 0 > data[i]:
            zerocrossing += 1
        if (data[i - 1]) < 0 < data[i]:
            zerocrossing += 1
    return zerocrossing
