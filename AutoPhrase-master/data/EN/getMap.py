fin = open("dict.en.txt", "r")
fout = open("en_map.txt", "w")

tnum = 4
for line in fin:
    val = line.strip().split(' ')
    fout.write("%d\t%s\n"%(tnum,val[0]))
    tnum = tnum + 1
fin.close()
fout.close()
