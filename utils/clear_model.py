import os

root = '../snapshots/'
nn = []
for f in os.listdir(root):
    for ff in os.listdir(root+f):
        dir_name = root+f
        for fff in os.listdir(dir_name):
            if fff =='opts.yaml' or fff=='GTA5_100000.pth':
                continue
            if fff.endswith('D1.pth') or fff.endswith('D2.pth'):
                    dst = dir_name+'/'+fff
                    print(dst)
                    os.remove(dst)

            try:
                if int(fff[5:10])//10000==1 and not fff.endswith('average.pth'):
                    dst = dir_name+'/'+fff
                    print(dst)
                    #os.remove(dst)
                if not(int(fff[5:10])%10000==0) and not fff.endswith('average.pth'):
                    dst = dir_name+'/'+fff
                    print(dst)
                    #os.remove(dst)
            except:
                continue

