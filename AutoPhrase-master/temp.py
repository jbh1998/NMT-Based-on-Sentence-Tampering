
fph = open("wmt14_phrase.en_t", "r")
fraw = open("data/EN/train.en", "r")
tnum = 0
for x,y in zip(fraw.readlines(), fph.readlines()):
    val = x.strip().split(' ')
    val2 = y.strip().split(' ')
    if len(val) != len(val2):
        print("%d raw %d phrase %d"%(tnum, len(val), len(val2)))

    tnum = tnum + 1

fph.close()
fraw.close()

