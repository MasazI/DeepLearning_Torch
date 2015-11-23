require 'gm'

-- shortcuts(looks like import of Python)
local tensor = torch.Tensor

do
    -- define graph
    nNodes = 10 -- ノードの数
    nStates = 2 -- ノードのとり得るステータス（離散）
    adjacency = gm.adjacency.full(nNodes) -- 隣接行列
    g = gm.graph{adjacency=adjacency, nStates=nStates, maxIter=10, verbose=true}
    
    -- unary potentials 単項のポテンシャル(ここでは全てのノードで既知という前提)
    nodePot = tensor{{1,3}, {9,1}, {1,3}, {9,1}, {1,1}, {1,3}, {9,1}, {1,3}, {9,1}, {1,1}}

    -- joint potentials
    edgePot = tensor(g.nEdges, nStates, nStates)
    basic = tensor{{2,1}, {1,2}}
    for e = 1, g.nEdges do
        edgePot[e] = basic
    end

    -- set potentials 便利な関数が用意されているunary potentialsとjoit potentialsを設定できる
    g:setPotentials(nodePot, edgePot)

    -- exact inference 推論の実行
    local exact = g:decode('exact')
    print()
    print('<gm.testme> exact optimal config:')
    print(exact)

    local nodeBel, edgeBel, logZ = g:infer('exact')
    print('<gm.testme> node beliefs:') 
    print(nodeBel)
    -- print('<gm.testme> edge beliefs:')
    -- print(edgeBel)   
    print('<gm.testme> log(Z)')
    print(logZ)

    -- belief propagation inference 確率伝播の推論 decodeをbpでやればよい、すごく簡単
    local optimal = g:decode('bp')
    print('<gm.testme> optimal config with belief propagation:')
    print(optimal)

    local nodeBel, edgeBel,logZ = g:infer('bp')
    print('<gm.testme> node beliefs:')
    print(nodeBel)
    print('<gm.testme> log(Z):')
    print(logZ)
end
