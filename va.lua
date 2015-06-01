require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'


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
exp_arg = nn.MulConstant(-1)(exp_arg)
exp = nn.Exp()(exp_arg)

sigma_inv = nn.Power(-1)(sigma)
l = {}
for i = 1, n_features do
  l[#l + 1] = nn.Select(2,i)(sigma_inv)
end
sigma_mm = nn.CMulTable()(l)
r = nn.CMulTable()({exp, sigma_mm})
r = nn.MulConstant(math.pow((2 * math.pi), -0.5 * n_features))(r)
r = nn.Log()(r)
loss_x = nn.MulConstant(-1)(r)
output = nn.Reshape(28, 28)(mu)
decoder = nn.gModule({z}, {output, loss_x})

trainset = mnist.traindataset()
testset = mnist.testdataset()
local n_data = 60000

local features_input = torch.zeros(n_data, 28, 28)
local e = torch.randn(n_data, 1)

for i = 1, n_data do
    features_input[{{i}, {}, {}}] = trainset[i].x:gt(125)
  
end


local params, grads = model_utils.combine_all_parameters(encoder, decoder)
local criterion = nn.MSECriterion()

-- return loss, grad
local feval = function(x)
  if x ~= params then
    params:copy(x)
  end
  grads:zero()

  --forward
  local z, loss_z = unpack(encoder:forward({features_input, e}))
  local output = decoder:forward(z)
  local loss_output = criterion:forward(output, features_input)
  local loss = torch.mean(loss_z) + loss_output
  --print(output[7]:gt(0.3))
  --print(features_input[7]:gt(0.5))  
  --print('--')
  print(loss_output)
  print(torch.mean(loss_z))
  
  --backward
  
  local doutput = criterion:backward(output, features_input)
  local dz = decoder:backward(z, doutput)
  local dloss_z = torch.ones(loss_z:size())
  encoder:backward({features_input, e}, {dz, dloss_z})

  return loss, grads
end

------------------------------------------------------------------------
-- optimization loop
--
local losses = {}
local optim_state = {learningRate = 1e-1}

for i = 1, 100 do
  local _, loss = optim.adagrad(feval, params, optim_state)
  losses[#losses + 1] = loss[1] -- append the new loss

  if i % 1 == 0 then
      print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))
      --print(params)
      
  end
end

z, loss_z = unpack(encoder:forward({features_input, e}))
output = decoder:forward(z)
print(output[9]:gt(0.4))
print(features_input[9]:gt(0.5))
print(output[10]:gt(0.4))
print(features_input[10]:gt(0.5))


a = 1




