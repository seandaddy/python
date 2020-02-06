# input_id = input("아이디를 입력해주세요.\n")
# members = ['egoing', 'k8805', 'leezche']
# for member in members:
#     if member == input_id:
#         print('Hello!, '+member)
#         import sys
#         sys.exit()
# print('Who are you?')

# in_str = input("Put your id.\n")
# real_egoing = "11"
# real_k8805 = "ab"
# if real_egoing == in_str:
#   print("Hello!, egoing")
# elif real_k8805 == in_str:
#   print("Hello!, k8805")
# else:
#   print("Who are you?")

input_id = input("아이디를 입력해주세요.\n")
def login(_id):
    members = ['egoing', 'k8805', 'leezche']
    for member in members:
        if member == _id:
            return True
    return False
if login(input_id):
    print('Hello, '+input_id)
else:
    print('Who are you?')
