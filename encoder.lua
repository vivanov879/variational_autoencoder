require 'mobdebug'.start()

require 'nn'
require 'nngraph'

n_features = 28 * 28
n_hidden = 500
n_z = 400

raw_features = nn.Identity()()
features = nn.Reshape(n_features)(raw_features)

h1 = nn.Linear(n_features, n_hidden)(features)
h2 = nn.Tanh()(h1)
mu = nn.Linear(n_hidden, n_z)(h2)
sigma = nn.Linear(n_hidden, n_z)(h2)
sigma = nn.Exp()(sigma)

e = nn.Identity()()
sigma_e = nn.CMulTable()({sigma, e})
z = nn.CAddTable()({mu, sigma_e})
mu_squared = nn.Square()(mu)
sigma_squared = nn.Square()(sigma)
log_sigma_sq = nn.Log()(sigma_squared)
minus_log_sigma = nn.MulConstant(-1)(log_sigma_sq)
loss_z = nn.CAddTable()({mu_squared, sigma_squared, minus_log_sigma})
loss_z = nn.AddConstant(-1)(loss_z)
loss_z = nn.MulConstant(0.5)(loss_z)
loss_z = nn.Sum(2)(loss_z)
encoder = nn.gModule({raw_features, e}, {z, loss_z})

return encoder