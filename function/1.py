def a3():
    print('aaa')
a3()

def a3():
    print('before')
    return 'aaa'
    print('after')
print(a3())

def a(num):
    return 'a'*num
print(a(3))

def make_string(str, num):
    return str*num
print(make_string('b', 3))
