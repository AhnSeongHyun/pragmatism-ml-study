f = open('./examples/korbit_btckrw.csv', 'r')

data = f.readlines()
f.close()

f = open('./examples/korbit_btckrw2.csv', 'w')
for i,d in enumerate(data):
    if i>=43202:
        f.write(d)

f.close()