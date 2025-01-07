


## Datamanager

## DimensionReducer

## 

class A:
    def __init__(self):
        self.count = 0


a = A()
print(a.count)

class B:
    def __init__(self):
        pass
    def add(self, a:A):
        a.count += 1
        return a
    


b = B()
a2 = b.add(a)

print(a2.count)