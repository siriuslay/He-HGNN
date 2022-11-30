import torch
import numpy as np
import torch.nn.functional as F


# alpha = torch.rand((2, 5))

# alpha = torch.FloatTensor([[0, 1, 0, -1, 1], [0, 1, 0, -1, 1]])
alpha = torch.tensor([[0, 0.8, 0.1, 0, 0.5], [0, 0.8, 0.1, 0, 0.5]])
a_s = F.softmax(alpha, dim=1)
a_log_s = F.log_softmax(alpha, dim=1)

tem = (torch.sum(alpha, dim=1) + 1e-8)
alpha_d = torch.diag_embed(torch.pow(tem, -1))
a = torch.mm(alpha_d, alpha)
print('a: ', a)

# a = F.softmax(alpha, dim=1)
a_log_a = torch.log(a + 1e-8)
# a_log_a = (torch.where(a > 0, (torch.log(a + 1e-8)).type(torch.float32), a))
print(torch.zeros_like(a))
# a_log_a_0 = (torch.where(a > 0, (torch.log(a + 1e-8)).type(torch.float32), torch.zeros_like(a)))

print('alpha: ', alpha)
print('a_log_s: ', a_log_s)
print('a_log_a: ', a_log_a)
# print('a_log_a_0: ', a_log_a_0)



def sample_gumbel(shape):
    """Sample from Gumbel(0, 1)"""
    U = np.random.uniform(size=shape)
    eps = np.finfo(U.dtype).eps
    r = -np.log(U + eps)
    r = -np.log(r + eps)
    return r

def gumbel_softmax_sample(prob, n, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    # eps = 1e-8
    # val = torch.zeros(prob.shape, dtype=torch.float32).to(self.dev)
    val = 0
    for i in range(n):
        r = torch.from_numpy(sample_gumbel(prob.shape))
        # values = (torch.where(prob > 0, (torch.log(prob + eps) + r).type(torch.float32), prob))
        # val += F.softmax((values / temperature), dim=1)
        value = F.softmax(((prob + r) / temperature), dim=1)
        print(f'value_{i}: ', value)
        val += value
    val = val / n
    # val = val + val.T.multiply(val.T > val) - val.multiply(val.T > val)
    rel = val.numpy()
    # val = F.normalize(val + torch.eye(val.shape[0]), dim=0, p=1).type(torch.float32)
    return val, rel

val, rel = gumbel_softmax_sample(a_log_a, 5, 0.2)
print('val: ', val)
print('rel: ', rel)

