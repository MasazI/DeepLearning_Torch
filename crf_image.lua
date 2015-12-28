require 'gm'
require 'image'

-- shortcuts
local tensor = torch.Tensor
local zeros = torch.zeros
local ones = torch.ones
local randn = torch.randn
local eye = torch.eye
local sort = torch.sort
local log = torch.log
local exp = torch.exp
local floor = torch.floor
local ceil = math.ceil
local uniform = torch.uniform


do
    -- load training data
    sample = torch.load('X.t7')
    print('sample shape:')
    print(sample:size())

    -- create image geometry
    nRows,nCols = sample:size(1),sample:size(2)
    nNodes = nRows*nCols
    nStates = 2

    -- how many instances to generate in the training data 
    nInstances=100

    -- make labels
    -- 元の画像を教師データにする。各画素の値がそのままlabelとなる。
    y = tensor(nInstances, nRows*nCols)
    for i = 1,nInstances do
        y[i] = sample
    end
    y = y + 1 -- lua is 1-based

    -- generate an ensemble of noisy versions of that image
    -- 訓練データを生成する。教師データにノイズを加えた画像
    X = tensor(nInstances,1,nRows*nCols)
    for i = 1,nInstances do
        X[i] = sample
    end
    X = X + randn(X:size())/2

    -- display a couple of input examples (can see in itorch notebook)
    -- いくつか教師データを表示する。これはitourch notebookで見ることができる。
    if itorch then
        itorch.image({X[1]:reshape(32,32), X[2]:reshape(32,32), X[3]:reshape(32,32), X[4]:reshape(32,32), X[5]:reshape(32,32)})
    end

    -- define adjacency matrix (4-connexity lattice)
    -- 隣接行列の定義(画像の場合は2次元=2dを生成する)
    local adj = gm.adjacency.lattice2d(nRows,nCols,4)
    
    -- create graph
    -- グラフの定義(グラフは隣接行列とノードのとり得るクラス数が必須)
    -- ここではtypeをcrfとしている。これはlossの計算方法を決定する。
    g = gm.graph{adjacency=adj, nStates=nStates, verbose=true, type='crf', maxIter=10}
    
    -- create node features ノードの特徴量を計算する。特徴量と重みを掛けあわせて使用する
    -- 2はとり得るクラス数
    Xnode = tensor(nInstances, 2, nNodes) -- 観測ノード
    -- bias項を最初のindexに追加(初期値1)
    Xnode[{ {}, 1 }] = 1 -- bias
    --normalize features:
    nFeatures = X:size(2) -- Xは(nInstances, 1, nRows*nCols)のサイズ def at l32、size(2)ではnFeatures=1が取得できる
    for f = 1,nFeatures do
        local Xf = X[{{}, f}]
        local mu = Xf:mean()
        local sigma = Xf:std()
        Xf:add(-mu):div(sigma)
    end
    Xnode[{ {} , 2}] = X -- features (simple normalized grayscale)
    nNodeFeatures = Xnode:size(2) -- Xnodeは(nInstances, 2, nNodes)のサイズ def at 56、size(2)ではnNodeFeatures=2が取得できる

    -- trainableなパラメータを決定する。tie node potentials to parameter vector
    -- 各確率変数間のポテンシャル関数を学習する。
    -- まずはnodeのポテンシャル関数f。
    nodeMap = zeros(nNodes, nStates, nNodeFeatures)
    for f = 1, nNodeFeatures do
        nodeMap[{ {}, 1, f }] = f
    end

    -- create edge features エッジの特徴量を計算する。
    nEdges = g.edgeEnds:size(1)
    nEdgeFeatures = nNodeFeatures*2 - 1 -- sharing bias(biasはEdgeごとには生成しない) nNodeFeaturesは2なので、ここは3になる
    Xedge = zeros(nInstances, nEdgeFeatures, nEdges)
    for i = 1,nInstances do -- instanceごとにループ
        for e=1,nEdges do -- edgeごとにループ
            local n1 = g.edgeEnds[e][1] -- edgeの先頭(便宜上先頭とか末尾とか呼ぶが無向グラフなので正確ではない)
            local n2 = g.edgeEnds[e][2] -- edgeの末尾
            for f = 1, nNodeFeatures do
                Xedge[i][f][e] = Xnode[i][f][n1] -- get all features form node1
            end
            for f = 1,nNodeFeatures-1 do
                Xedge[i][nNodeFeatures+f][e] = Xnode[i][f+1][n2] -- get all features fron node1, except bias(shared)
            end
        end
    end

    -- つぎにedgesのポテンシャル関数ef。
    local f = nodeMap:max()
    edgeMap = zeros(nEdges, nStates, nStates, nEdgeFeatures) -- edgeMapのポテンシャル関数のMapはエッジとその両端がとり得るクラス数、エッジの特徴量をもつ行列
    for ef = 1, nEdgeFeatures do
        edgeMap[{ {}, 1, 1, ef }] = f+ef
        edgeMap[{ {}, 2, 2, ef }] = f+ef
    end

    -- 学習パラメータの初期化
    g:initParameters(nodeMap, edgeMap)

    -- 学習
    -- 学習率は0.001
    require 'optim'
    local sgdState = {
        learningRate = 1e-3,
        learningRateDecay = 1e-2,
        weightDecay = 1e-5,
    }
    for iter = 1,100 do
        -- SGD step:
        optim.sgd(function()
            -- random sample:
            local i = torch.random(1,nInstances)
            -- compute f+grad:
            local f,grad = g:nll('bp',y[i],Xnode[i],Xedge[i])
            -- verbose:
            print('SGD @ iteration ' .. iter .. ': objective = ', f)
            -- return f+grad:
            return f,grad
        end, 
        g.w, sgdState)
    end

    -- the model is trained, generate node/edge potentials, and test
    marginals = {}
    labelings = {}
    for i = 1,4 do
        g:makePotentials(Xnode[i],Xedge[i])
        nodeBel = g:infer('bp')
        labeling = g:decode('bp')
        table.insert(marginals,nodeBel[{ {},2 }]:reshape(nRows,nCols))
        table.insert(labelings,labeling:reshape(nRows,nCols))
    end
end
