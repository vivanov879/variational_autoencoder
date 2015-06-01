require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'


local encoder = require 'Encoder'
local decoder = require 'Decoder'


trainset = mnist.traindataset()
testset = mnist.testdataset()
local n_data = 1000

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
  local z, loss_z = unpack(encoder:forward({features_input, e}))
  local output, loss_x = unpack(decoder:forward({features_input, z}))
  local loss = torch.mean(loss_z) + torch.mean(loss_x)
  --print(output[7]:gt(0.3))
  --print(features_input[7]:gt(0.5))  
  --print('--')
  print(torch.mean(loss_x))
  print(torch.mean(loss_z))
  
  --backward
  
  dfeatures_input1, dz = unpack(decoder:backward({features_input, z}, {torch.zeros(output:size()), torch.ones(loss_x:size())}))
  dfeatures_input2, de = unpack(encoder:backward({features_input, e}, {dz, torch.ones(loss_z:size())}))

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
output, _ = unpack(decoder:forward({torch.zeros(features_input:size()), z}))
print(output[9]:gt(0.6))
print(features_input[9]:gt(0.5))
print(output[10]:gt(0.6))
print(features_input[10]:gt(0.5))


a = 1




