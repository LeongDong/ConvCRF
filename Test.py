import torch
import torch.nn.functional as F

if __name__=='__main__':
    a = torch.randn(2,3)
    print(a)
    b = F.softmax(a,dim=0)
    print(b)