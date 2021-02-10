def foo(a, *args):
    print(a, args)
    x, y = args
    print(x, y)

foo(3, 1, 2)