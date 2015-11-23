require 'gm'
require 'image'

-- shortcut
local tensor = torch.Tensor
local randn = torch.randn
local zeros = torch.zeros

do
    -- load training data
    sample = torch.load('X.t7')
    print(sample:size())

    -- create image geometry
    nRows,nCols = sample:size(1),sample:size(2)
    nNodes = nRows*nCols

    -- how many instances to generate in the training data 
    nInstances=100

    -- make labels
    y = tensor(nInstances, nRows*nCols)
    for i = 1,nInstances do
        y[i] = sample
    end
    y = y + 1 -- lua is 1-based

    -- generate an ensemble of noisy versions of that image
    X = tensor(nInstances,1,nRows*nCols)
    for i = 1,nInstances do
        X[i] = sample
    end
    X = X + randn(X:size())/2

    -- display a couple of input examples (can see in itorch notebook)
    if itorch then
        itorch.image({X[1]:reshape(32,32), X[2]:reshape(32,32), X[3]:reshape(32,32), X[4]:reshape(32,32), X[5]:reshape(32,32)})
    end

    -- define adjacency matrix (4-connexity lattice)
    local adj = gm.adjacency.lattice2d(nRows,nCols,4)

    -- create graph
    nStates = 2
    g = gm.graph{adjacency=adj, nSates=nStates, verbose=true, type='crf', maxIter=10}
    
    -- create node features 観測ノード (normalized X and a bias)
    Xnode = tensor(nInstances, 2, nNodes) -- 観測ノード
    Xnode[{{}, 1}] = 1 -- bias

    --normalize features:
    nFeatures = X:size(2) -- Xは(nInstances, 1, nRows*nCols)のサイズ、size(2)では1が取得できる
    print(nFeatures)

    for f = 1,nFeatures do
        local Xf = X[{{}, f}]
        local mu = Xf:mean()
        local sigma = Xf:std()
        Xf:add(-mu):div(sigma)
    end
    Xnode[{{}, 2}] = X -- features (simple normalized grayscale)
    nNodeFeatures = Xnode:size(2)
    print(nNodeFeatures)

    -- tie node potentials to parameter vector
    nodeMap = zeros(nNodes, nStates, nNodeFeatures)
    for f = 1, nNodeFeatures do
        nodeMap[{{}, 1, f}] = f
    end



end
