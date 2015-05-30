require 'mobdebug'.start()

require 'nn'
require 'nngraph'


x = nn.Identity()()
y = nn.Reshape(2, 3)(x)
z = nn.Select(3,3)(y)
q = nn.Reshape(6)(y)
m = nn.gModule({x},{q})
a = torch.rand(10,6)

print(m:forward(a))
print(a)



a = 1


