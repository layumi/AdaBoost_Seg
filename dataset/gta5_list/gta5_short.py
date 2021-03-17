fp  = open('train_short.txt', 'w')
index = 0 
with open('train.txt') as f: 
    for line in f:
        index = index + 1
        if index%100 == 0:
            fp.write(line)
