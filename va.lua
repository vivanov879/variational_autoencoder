require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'

n_features = 28 * 28
n_hidden = 200
n_z = 20

--encoder
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

-- decoder
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


trainset = mnist.traindataset()
testset = mnist.testdataset()
local n_data = 100

features_input = torch.zeros(n_data, 28, 28)
e = torch.randn(n_data, n_z)

for i = 1, n_data do
    features_input[{{i}, {}, {}}] = trainset[i].x:gt(125)
  
end


params, grads = model_utils.combine_all_parameters(encoder, decoder)
criterion = nn.MSECriterion()

-- return loss, grad
local feval = function(x)
  if x ~= params then
    params:copy(x)
  end
  grads:zero()

  --forward
  z, loss_z = unpack(encoder:forward({features_input, e}))
  local output, loss_x = unpack(decoder:forward({features_input, z}))
  local loss = torch.mean(loss_z) + torch.mean(loss_x)
  --print(output[7]:gt(0.3))
  --print(features_input[7]:gt(0.5))  
  --print('--')
  --print(torch.mean(loss_x))
  --print(torch.mean(loss_z))
  
  --backward
  
  dfeatures_input1, dz = unpack(decoder:backward({features_input, z}, {torch.zeros(output:size()), torch.ones(loss_x:size())}))
  dfeatures_input2, de = unpack(encoder:backward({features_input, e}, {dz, torch.ones(loss_z:size())}))

  return loss, grads
end

------------------------------------------------------------------------
-- optimization loop
--
optim_state = {learningRate = 1e-2}

for i = 1, 1000 do
  local _, loss = optim.adagrad(feval, params, optim_state)

  if i % 10 == 0 then
      print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))
      --print(params)
      
  end
end

z, loss_z = unpack(encoder:forward({features_input, e}))
output, _ = unpack(decoder:forward({torch.zeros(features_input:size()), z}))
print(output[9]:gt(0.5))
print(features_input[9]:gt(0.5))
print(output[10]:gt(0.5))
print(features_input[10]:gt(0.5))


a = 1




