require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'


local mnist = require 'mnist'

trainset = mnist.traindataset()
testset = mnist.testdataset()

print(trainset.size) -- to retrieve the size
print(testset.size) -- to retrieve the size

i = 100
local ex = trainset[i]
local x = ex.x -- the input (a 28x28 ByteTensor)
local y = ex.y -- the label (0--9)

local x = torch.zeros(100, 28, 28)
for i = 1, 100 do 
  
  x[{{i}, {}, {}}] = trainset[i].x
  
  
end

binarized_x = x:gt(125)

image.display(x)
image.display(binarized_x)




