using Revise

using SparseArrays
using ProgressMeter

function potjans_params()
    conn_probs = [[0.1009,  0.1689, 0.0437, 0.0818, 0.0323, 0.,     0.0076, 0.    ],
                [0.1346,   0.1371, 0.0316, 0.0515, 0.0755, 0.,     0.0042, 0.    ],
                [0.0077,   0.0059, 0.0497, 0.135,  0.0067, 0.0003, 0.0453, 0.    ],
                [0.0691,   0.0029, 0.0794, 0.1597, 0.0033, 0.,     0.1057, 0.    ],
                [0.1004,   0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.    ],
                [0.0548,   0.0269, 0.0257, 0.0022, 0.06,   0.3158, 0.0086, 0.    ],
                [0.0156,   0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
                [0.0364,   0.001,  0.0034, 0.0005, 0.0277, 0.008,  0.0658, 0.1443]]

    columns_conn_probs = [col for col in eachcol(conn_probs)][1]
    #@show(columns_conn_probs)
    layer_names = ["23E","23I","4E","4I","5E", "5I", "6E", "6I"]
    #cell_counts = Dict("23E"=>20683, "23I"=>5834,
    #            "4E"=>21915, "4I"=>5479,
    #            "5E"=>4850, "5I"=>1065,
    #            "6E"=>14395, "6I"=>2948)
    ccuf = Dict(
        k=>v for (k,v) in zip(layer_names,columns_conn_probs)
    )

    ccu = Dict("23E"=>20683, "23I"=>5834,
                "4E"=>21915, "4I"=>5479,
                "5E"=>4850, "5I"=>1065,
                "6E"=>14395, "6I"=>2948)

    #c = ceil(Int64, a/b)
    ccu = Dict((k,ceil(Int64,v/35.0)) for (k,v) in pairs(ccu))
    cumulative = Dict() 
    v_old=1
    for (k,v) in pairs(ccu)
        cumulative[k]=collect(v_old:v+v_old)
        v_old=v+v_old
    end
    return (cumulative,ccu,ccuf,layer_names,columns_conn_probs,conn_probs)
end

function genWeights(p)

    nc0Max = Int(p.Ncells*p.pree) # outdegree
    nc0 = Int.(nc0Max*ones(p.Ncells))
    w0Index = spzeros(Int,nc0Max,p.Ncells)
    w0Weights = spzeros(nc0Max,p.Ncells)
    for i = 1:p.Ncells
        postcells = filter(x->x!=i, collect(1:p.Ncells)) # remove autapse
        ###
        # take a small random subset of all possible post synaptic cells
        # by randomly shuffling all of the cell indexs, and then taking a small finite slice of the subset.
        ###
        temp = sort(shuffle(postcells)[1:nc0Max]) # fixed outdegree nc0Max
        w0Index[1:nc0Max,i] = temp
        nexc = sum(w0Index[1:nc0Max,i] .<= p.Ne) # number of exc synapses
        if i <= p.Ne
            w0Weights[1:nexc,i] .= p.jee  ## EE weights
            w0Weights[nexc+1:nc0Max,i] .= p.jie  ## IE weights
        else
            w0Weights[1:nexc,i] .= p.jei  ## EI weights
            w0Weights[nexc+1:nc0Max,i] .= p.jii  ## II weights
        end
    end

    return w0Index, w0Weights, nc0
end
#using ProgressMeters

function re_write_weights(Ncells)
    (cumulative,ccu,ccuf,layer_names,columns_conn_probs,conn_probs) = potjans_params()    
    w_mean = 87.8e-3  # nA
    ###
    # Efficiency right!
    # Lower memory footprint motivations.
    # A 2D matrix should be stored as 1D matrix of srcs,tgts
    # A weight matrix should be stored as 1 matrix, which is redistributed in loops using 
    # the 1D matrix of srcs,tgts.
    ###
    #@show(cumulative)
    #@show(values(cumulative))

    Ncells = sum([i for i in values(ccu)])+1#max([max(i[:]) for i in values(cumulative)])
    nc0Max = Ncells-1 # outdegree
    nc0 = Int.(nc0Max*ones(Ncells))

    w0Index = spzeros(Int,Ncells,Ncells)
    w0Weights = spzeros(Float32,Ncells,Ncells)
    @showprogress for (i,(k,v)) in enumerate(pairs(cumulative))
        for src in v
            for (j,(k1,v1)) in enumerate(pairs(cumulative))
                for tgt in v1
                    if src!=tgt
                        prob = conn_probs[i][j]
                        if rand()<prob
                            if occursin("E",k)                     
                                w0Weights[tgt,src] = w_mean                                
                            end
                            if occursin("I",k)
                                w0Weights[tgt,src] = -w_mean
                            end
                            w0Index[tgt,src] = tgt
                        end
                    end
                end
            end
        end
    end
    return (w0Index,w0Index,nc0)
end
    #=
        #postcells = filter(x->x!=i, collect(1:p.Ncells)) # remove autapse
        w0Index[1:nc0Max,i] = sort(shuffle(postcells)[1:nc0Max]) # fixed outdegree nc0Max
        #@show(w0Index[1:nc0Max,i])
        nexc = sum(w0Index[1:nc0Max,i] .<= p.Ne) # number of exc synapses
        if i <= p.Ne
            w0Weights[1:nexc,i] .= w_mean  ## EE weights
            w0Weights[nexc+1:nc0Max,i] .= p.jie  ## IE weights
        else
            w0Weights[1:nexc,i] .= p.jei  ## EI weights
            w0Weights[nexc+1:nc0Max,i] .= p.jii  ## II weights
        end
    end
    =#

function genWeights_square(p)
    w_mean = 87.8e-3  # nA

    nc0Max = Int(p.Ncells*p.pree) # outdegree
    nc0 = Int.(p.Ncells*ones(p.Ncells))
    w0Index = spzeros(Int,p.Ncells,p.Ncells)
    w0Weights = spzeros(p.Ncells,p.Ncells)
    for i = 1:p.Ncells
        postcells = filter(x->x!=i, collect(1:p.Ncells)) # remove autapse
        w0Index[1:nc0Max,i] = sort(shuffle(postcells)[1:nc0Max]) # fixed outdegree nc0Max
        #@show(w0Index[1:nc0Max,i])
        nexc = sum(w0Index[1:nc0Max,i] .<= p.Ne) # number of exc synapses
        if i <= p.Ne
            w0Weights[1:nexc,i] .= w_mean  ## EE weights
            w0Weights[nexc+1:nc0Max,i] .= p.jie  ## IE weights
        else
            w0Weights[1:nexc,i] .= p.jei  ## EI weights
            w0Weights[nexc+1:nc0Max,i] .= p.jii  ## II weights
        end
    end
    
    return w0Index, w0Weights, nc0
    
    
    end

