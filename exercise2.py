in_id = input("아이디를 입력해주세요.\n")
in_pwd = input("비밀번호를 입력해주세요.\n")
real_id = "egoing"
real_pwd = "11"
if real_id == in_id and real_pwd == in_pwd:
    print("Hello!")

elif real_id == in_id:
    print("잘못된 비밀번호입니다")
else:
    print("잘못된 아이디입니다")