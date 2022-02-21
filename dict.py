from turtle import update
from cv2 import reduce
import torch

class Dictionary:
    def __init__(self, size=(200,8100),l=0.5, gamma = 0.005) -> None:
        self.D = torch.rand(size)
        self.l = l
        self.gamma = gamma

    def getError(self, X:torch.Tensor, A:torch.Tensor)->float:
        assert(X.shape[0] == A.shape[0])
        assert(A.shape[1] == self.D.shape[0])
        assert(self.D.shape[1] == X.shape[1])
        # X \in s*f
        # D \in d*f
        # A \in s*d 

        segLenth = X.shape[0]
        Tf = (X - A@self.D)
        fx = 0.5*(1/segLenth)*torch.sum(torch.linalg.norm(Tf, axis = 1)*torch.linalg.norm(Tf, axis = 1))
        gx = self.l * torch.sum(torch.linalg.norm(A, axis = 1)*torch.linalg.norm(A, axis = 1))
        return fx+gx

    # def reduceDict(self, X, lr=1.0):
    #     # 使用这个sb的减梯度，根本收敛不下去， 还必须把二范数平方改为二范数
    #     s = X.shape[0]
    #     d = self.D.shape[0]
    #     A = torch.rand((s,d),requires_grad=True)
    #     # 优化只会减梯度
    #     for i in range(20):
    #         error = self.getError(X, A)
    #         error.backward()
    #         A.data.add_(A.grad, alpha=-lr)
    #         A.grad = None

    #     A.requires_grad = False
    #     return A, self.getError(X, A)


        

    # def updateDict(self, X, lr=1.0):
    #     m = len(X)
    #     self.D.requires_grad = True
    #     for i in range(10):
    #         error = 0.0
    #         self.D.data.grad = None
    #         for (x, a) in enumerate(X):
    #             error = error + 1/float(m) * self.getError(x, a)
    #         error += self.gamma* torch.sum(torch.linalg.norm(self.D, axis = 1))
    #         error.backward()
    #         self.D.data.add_(self.D.grad, alpha=-lr)
    #     self.D.requires_grad = False

    # def initDict(self, X, lr=1.0):
    #     A_set = []
    #     m = len(X)
    #     for x in X:
    #         A, _ = self.reduceDict(x,0.1)
    #         A_set.append(A)
    #     self.D.requires_grad = True
    #     for i in range(20):
    #         error = 0.0
    #         self.D.data.grad = None
    #         for (i, x) in enumerate(X):
    #             error = error + 1/float(m) * self.getError(x, A_set[i])
    #         print(f"reduce each A: {error}")
    #         error += self.gamma* torch.sum(torch.linalg.norm(self.D, axis = 1))
    #         error.backward()
    #         print(error)
    #         self.D.data.add_(self.D.grad, alpha=-lr)
    #     self.D.requires_grad = False

    def reduceDict(self, X, lr=1.0):
        DT = self.D.transpose(0,1)
        s = X.shape[0]
        d = self.D.shape[0]
        A = X @ DT @ torch.linalg.inv(2 * self.l * s * torch.eye(d) + (self.D @ DT) )
        return A, self.getError(X, A)

    def updateDict(self, X, lr=1.0):
        m = 10
        s = X[0][0].shape[0]
        d = self.D.shape[0]
        temax = 0.0
        temaa = self.gamma*2*torch.eye(d)
        for x, a in X[-10:]:
            A,_= self.reduceDict(x)
            AT = A.transpose(0,1)
            temax += 1/float(m*s)*AT@x
            temaa += 1/float(m*s)*AT@A
        self.D = torch.linalg.inv(temaa)@temax

    def initDict(self, X, lr=1.0):
        A_set = []
        m = len(X)
        s = X[0].shape[0]
        d = self.D.shape[0]
        for x in X:
            A, error = self.reduceDict(x,0.1)
            A_set.append(A)
        temax = 0.0
        temaa = self.gamma*2*torch.eye(d)
        for (i, x) in enumerate(X):
            A = A_set[i]
            AT = A.transpose(0,1)
            temax += 1/float(m*s)*AT@x
            temaa += 1/float(m*s)*AT@A
        self.D = torch.linalg.inv(temaa)@temax 

        
if __name__ == "__main__":
    data = []
    # N*S*F  N*50*8100 
    size = (50,8100)
    for i in range(100):
        data.append(torch.ones(size))

    it = torch.zeros((50,8100))
    it = it + torch.sin(torch.arange(0,8100))
    for i in range(100):
        data.append(it)

    for i in range(100):
        data.append(torch.ones(size))

    print("get the data...")
    startNum = 30
    std =2
    X_ = []
    myDict = Dictionary()
    Chosen = []
    res = []
    cnt = 0
    for i,d in enumerate(data):
        if i < startNum :
            X_.append(d)
            Chosen.append(i)
            continue
        if i == startNum:
            print("init...")
            myDict.initDict(X_, 1)
        print(i)
        A, error = myDict.reduceDict(d, 0.1)
        print(error)
        if error < std:
            cnt+=1
            continue
        else:
            print(f"update:{i}")
            Chosen.append(i)
            res.append((d,A))
            myDict.updateDict(res, 0.1)
    print(Chosen)
    print(cnt)  