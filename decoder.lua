require 'mobdebug'.start()

require 'nn'
require 'nngraph'


n_features = 28 * 28
n_hidden = 500
n_z = 10


raw_features = nn.Identity()()
features = nn.Reshape(n_features)(raw_features)
z = nn.Identity()()
h1 = nn.Linear(n_z, n_hidden)(z)
h2 = nn.Tanh()(h1)
mu = nn.Linear(n_hidden, n_features)(h2)
sigma = nn.Linear(n_hidden, n_features)(h2)
sigma = nn.Exp()(sigma)

neg_mu = nn.MulConstant(-1)(mu)
d = nn.CAddTable()({features, neg_mu})
d2 = nn.Power(2)(d)
sigma2_inv = nn.Power(-2)(sigma)
exp_arg = nn.CMulTable()({d2, sigma2_inv})
exp_arg = nn.Sum(2)(exp_arg)

sigma_mm = nn.Log()(sigma)
sigma_mm = nn.Sum(2)(sigma)
loss_x = nn.CAddTable()({exp_arg, sigma_mm})
loss_x = nn.AddConstant(0.5 * n_features * math.log((2 * math.pi)))(loss_x)

output = nn.Reshape(28, 28)(mu)
decoder = nn.gModule({raw_features, z}, {output, loss_x})

return decoder
