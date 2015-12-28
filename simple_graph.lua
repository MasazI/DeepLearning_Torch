require 'gm'

-- shortcuts(looks like import of Python)
local tensor = torch.Tensor

do
    -- 問題設定
    -- 非常にシンプルなグラフを用いてノードのラベルを推定する
    -- ラベルは2値である。[0, 1]
    -- 各ノードのラベルごとのポテンシャルは既知とする
    -- エッジ(Factor)のjointポテンシャルは既知とする
    -- 

    -- define graph
    nNodes = 2 -- ノードの数
    nStates = 2 -- ノードのとり得るステータス（離散）
    adjacency = gm.adjacency.full(nNodes) -- 隣接行列
    g = gm.graph{adjacency=adjacency, nStates=nStates, maxIter=1, verbose=true}
    
    -- unary potentials 単項のポテンシャル(ここでは全てのノード(10個=nNodes)で既知という前提)
    -- nodePot = tensor{{1,3}, {9,1}, {1,3}, {9,1}, {1,1}, {1,3}, {9,1}, {1,3}, {9,1}, {1,1}}
    nodePot = tensor{{1,9}, {9,1}}

    -- joint potentials 同時分布のポテンシャル(ここでは隣接ノードの関係が既知という前提)
    edgePot = tensor(g.nEdges, nStates, nStates)
    basic = tensor{{1,1}, {1,0}}
    -- 各edgeに上記ポテンシャルを設定する
    -- 各ノードは2つのエッジをもつのでtensorの要素は2つ
    --
    print('edge num:')
    print(g.nEdges) 
    for e = 1, g.nEdges do
        edgePot[e] = basic
    end

    -- set potentials 便利な関数が用意されている(unary potentialsとjoit potentialsをグラフに設定できる)
    g:setPotentials(nodePot, edgePot)

    -- exact inference 推論の実行(厳密な推論)
    local exact = g:decode('exact')
    print()
    print('<gm.testme> exact optimal config:')
    print(exact)

    local nodeBel, edgeBel, logZ = g:infer('exact')
    print('<gm.testme> node beliefs:') 
    print(nodeBel)
    print('<gm.testme> edge beliefs:')
    print(edgeBel)
    print('<gm.testme> log(Z)')
    print(logZ)

    -- belief propagation inference 確率伝播の推論 decodeをbpでやればよい、すごく簡単に実行可能
    local optimal = g:decode('bp')
    print('<gm.testme> optimal config with belief propagation:')
    print(optimal)

    local nodeBel, edgeBel,logZ = g:infer('bp')
    print('<gm.testme> node beliefs:')
    print(nodeBel)
    print('<gm.testme> edge beliefs:')
    print(edgeBel)
    print('<gm.testme> log(Z):')
    print(logZ)
end
