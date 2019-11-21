import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import os
import torch.utils.model_zoo as model_zoo

class LambdaBase(nn.Sequential):
    def __init__(self, *args):
        super(LambdaBase, self).__init__(*args)

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def __init__(self, *args):
        super(Lambda, self).__init__(*args)
        self.lambda_func = identity

    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def __init__(self, *args):
        super(LambdaMap, self).__init__(*args)
        self.lambda_func = identity

    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def __init__(self, *args):
        super(LambdaReduce, self).__init__(*args)
        self.lambda_func = add

    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

def identity(x): return x

def add(x, y): return x + y

resnext101_64x4d_features = nn.Sequential(#Sequential,
    nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias = False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d((3, 3), (2, 2), (1, 1)),
    nn.Sequential(#Sequential,
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(256),
                ),
                nn.Sequential(#Sequential,
                    nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(256),
                ),
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(256),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(256),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential(#Sequential,
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(512),
                ),
                nn.Sequential(#Sequential,
                    nn.Conv2d(256, 512, (1, 1), (2, 2), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(512),
                ),
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential(#Sequential,
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (2, 2), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                nn.Sequential(#Sequential,
                    nn.Conv2d(512, 1024, (1, 1), (2, 2), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential(#Sequential,
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                        nn.Conv2d(2048, 2048, (3, 3), (2, 2), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(2048),
                ),
                nn.Sequential(#Sequential,
                    nn.Conv2d(1024, 2048, (1, 1), (2, 2), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(2048),
                ),
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                        nn.Conv2d(2048, 2048, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(2048),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                        nn.Conv2d(2048, 2048, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(2048),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
    )
)

pretrained_settings = {
    'resnext101_64x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_64x4d-e77a0586.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    }
}

class ResNeXt101_64x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_64x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_64x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

def resnext101_64x4d(num_classes=1000, pretrained='imagenet'):
    model = ResNeXt101_64x4d(num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnext101_64x4d'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model