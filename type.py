print(type('egoing')) #<class 'str'>
name = 'egoing'
print(name) #egoing
print(type(['egoing', 'leezche', 'graphittie'])) #<class 'list'>
names = ['egoing', 'leezche', 'graphittie']
print(names)
print(names[2]) #graphittie
egoing = ['programmer', 'seoul', 25, False]
egoing[1] = 'busan'
print(egoing) #['programmer', 'busan', 25, False]

al = ['A', 'B', 'C', 'D']
print(len(al)) # 4
al.append('E')
print(al) #['A', 'B', 'C', 'D', 'E']
del(al[0])
print(al) #['B', 'C', 'D', 'E']

print("Hello world 0")
print("Hello world 9")
print("Hello world 18")
print("Hello world 27")
print("Hello world 36")
print("Hello world 45")
print("Hello world 54")
print("Hello world 63")
print("Hello world 72")
print("Hello world 81")

i = 0
while i < 10:
    print("Hello world "+str(i*9))
    i = i + 1

i = 0
while i < 10:
    if i == 4:
        break
    print(i)
    i = i + 1
print('after while')
