using SparseArrays
function genPlasticWeights(p, w0Index, nc0, ns0, matchedCells)
    #@show(w0Index)
    # rearrange initial weights
    w0 = Dict{Int,Array{Int,1}}()
    for i = 1:p.Ncells
        w0[i] = []
    end
    for preCell = 1:p.Ncells
        for i = 1:nc0[preCell]
            postCell = w0Index[i,preCell]+1
            push!(w0[postCell],preCell)
        end
    end

    exc_selected = collect(1:p.Ncells)
    inh_selected = collect(1:p.Ncells)
    
    # define weights_plastic    
    wpWeightIn = spzeros(p.Ncells,round(Int,2*p.L))
    wpIndexIn = spzeros(p.Ncells,round(Int,2*p.L))
    @show(length(exc_selected))
    @show(p.L)

    indE = sort(shuffle(exc_selected)[1:p.L])
    indI = sort(shuffle(inh_selected)[1:p.L])
    ncpIn = zeros(Int,p.Ncells)

    # random plastic weights
    for postCell = 1:p.Ncells
        # select random neurons
        matchedCells_noautapse = filter(x->x!=postCell, matchedCells)
        indE = sort(shuffle(matchedCells_noautapse)[1:p.L])
        indI = sort(shuffle(inh_selected)[1:p.L])
        #indE = sort(shuffle(exc_selected)[1:ceil(Int,p.Ncells/2)])
        #indI = sort(shuffle(inh_selected)[1:ceil(Int,p.Ncells/2)])

        # build wpIndexIn
        ind = [indE; indI]
        wpIndexIn[postCell,:] = ind
        ncpIn[postCell] = length(ind)
    end
    #=
    # trained exc neurons form a cluster
    for ii = 1:length(matchedCells)
        # neuron to be trained
        postCell = matchedCells[ii]

        # # (1) select presynaptic neurons with similar firing rates
        # #       (a) exc synapses from other trained exc neurons; inh synapses are random
        # #       (b) select consecutive neurons from matchedCells: connect trained neurons with similar firing rates
        # matchedCells_noautapse = filter(x->x!=postCell, matchedCells)
        # leftInd = maximum([1, round(Int, ii-p.Lexc/2)])
        # rightInd = leftInd + p.Lexc - 1
        # if rightInd > length(matchedCells_noautapse)
        #     leftInd = length(matchedCells_noautapse) - p.Lexc + 1
        #     rightInd = length(matchedCells_noautapse)
        # end
        # indE = sort(matchedCells_noautapse[leftInd:rightInd])
        # indI = sort(shuffle(inh_selected)[1:p.L])

        # (2) excitatory: select random but trained neurons as presynaptic neurons
        #     inhibitory: select random presynaptic neurons
        #matchedCells_noautapse = filter(x->x!=postCell, matchedCells)
        indE = sort(shuffle(matchedCells_noautapse)[1:ceil(Int,p.L)-1])
        indI = sort(shuffle(inh_selected)[1:ceil(Int,p.L)])

        # updated wpIndexIn for postcell in matchedCells
        ind = [indE; indI]
        wpIndexIn[postCell,:] = ind
        ncpIn[postCell] = length(ind)

        # (1) update plastic weights to postcell in matchedCells
        # (2) other plastic weights = 0 
        if postCell <= p.Ne
            wpee = p.wpee*ones(p.L)
            wpei = p.wpei*ones(p.L)
            #wpee = p.wpee*ones(floor(Int,p.Ncells/2))
            #wpei = p.wpei*ones(floor(Int,p.Ncells/2))
            wpWeightIn[postCell,:] = [wpee; wpei]
        else
            wpie = p.wpie*ones(p.Lexc)
            wpii = p.wpii*ones(p.Linh)
            #wpie = p.wpie*ones(floor(Int,p.Ncells/2))
            #wpii = p.wpii*ones(floor(Int,p.Ncells/2))
            wpWeightIn[postCell,:] = [wpie; wpii]
        end
    end
    =#
    # define feedforward weights to excitatory neurons
    # wpWeightFfwd = randn(p.Ne, p.Lffwd) * p.wpffwd
    wpWeightFfwd = Vector{Array{Float64,2}}(); 
    for licki = 1:2
        # wtmp = zeros(p.Ne, p.Lffwd)
        wtmp = randn(p.Ne, p.Lffwd) * p.wpffwd
        # for ii = 1:length(matchedCells)
        #     postCell = matchedCells[ii]
        #     wtmp[postCell,:] = wpee*randn(p.Lffwd)
        # end
        push!(wpWeightFfwd, wtmp)
    end

    # get indices of postsynaptic cells for each presynaptic cell
    wpIndexConvert = zeros(p.Ncells,round(Int,p.Lexc+p.Linh))
    wpIndexOutD = Dict{Int,Array{Int,1}}()
    ncpOut = zeros(Int,p.Ncells)
    for i = 1:p.Ncells
        wpIndexOutD[i] = []
    end
    for postCell = 1:p.Ncells
        for i = 1:ncpIn[postCell]
            preCell = wpIndexIn[postCell,i]
            push!(wpIndexOutD[preCell],postCell)
            wpIndexConvert[postCell,i] = length(wpIndexOutD[preCell])
        end
    end
    for preCell = 1:p.Ncells
        ncpOut[preCell] = length(wpIndexOutD[preCell])
    end

    # get weight, index of outgoing connections
    ncpOutMax = Int(maximum(ncpOut))
    wpIndexOut = zeros(ncpOutMax,p.Ncells)
    wpWeightOut = zeros(ncpOutMax,p.Ncells)
    for preCell = 1:p.Ncells
        wpIndexOut[1:ncpOut[preCell],preCell] = wpIndexOutD[preCell]
    end
    wpWeightOut = convertWgtIn2Out(p,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)
    
    return wpWeightFfwd, wpWeightIn, wpWeightOut, wpIndexIn, wpIndexOut, wpIndexConvert, ncpIn, ncpOut
    
end
