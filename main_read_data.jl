using Distributions
using Plots
using OhMyREPL
#using PyCall
#using PyPlot
using DelimitedFiles
using LinearAlgebra
using Random
using SparseArrays
using JLD
using Revise
#unicodeplots()
# modifiable parameters
lam_list = [0.5, 1.0]
jj = 2

L_list = [6.0, 12.0]
ll = 1

Lffwd_list = [100, 150, 200, 250, 300, 350, 400]
oo = 5

wpffwd_list = [0.0, 1.0, 2.0]
pp = 1

fracTrained_list = vcat(collect(0.1:0.1:0.7), 1824/2500)
qq = 8 # Ntrained = Npyr
@show(fracTrained_list)

include("param.jl")
include("genWeights.jl")
include("genPlasticWeights.jl")
include("convertWgtIn2Out.jl")
include("genffwdRate.jl")
include("genStim.jl")
include("genCellsTrained.jl")
include("runinitial.jl")
include("runtrain.jl")
include("runtest.jl")
include("funMovAvg.jl")
include("funSample.jl")
include("funRollingAvg.jl")
include("runperformance.jl")
include("plot.jl")

dirNetwork = "data_network/"
dirALM = "data_alm/"

#----------- initialization --------------#

# run initial balanced network

w0Index, w0Weights, nc0, w0Index_,Ne, Ni = convert_dense_matrices(p)
###???
###????
p.Ni = Ni
p.Ne = Ne

if !isfile("potjans_weights.jld")
    (edge_dict,w0Weights,w0Index_,Ne,Ni) = re_write_weights(p.Ncells)
    save("potjans_weights.jld", "w0Weights", w0Weights, "w0Index", w0Index,"nc0",nc0)
end
if isfile("potjans_weights.jld")
   @load "potjans_weights.jld"
end

w0Weights = sparse(w0Weights)
w0Index = sparse(w0Index)
@show(w0Weights)
@show(w0Index)
postcells = filter(x->x!=0, w0Weights)
@show(size(postcells))


#@time uavg, ns0, ustd = runinitial(p,w0Index, w0Weights, nc0)

function plot_spikes(ns0,times,filename)
    tt = convert(Array{Float64,2},times[ns0])
    println("Number of spikes")
    println(length(tt[tt .> 0]))
    tt = tt[tt .> 0]
    whole_mat = reduce(vcat, tt)
    whole_mat = whole_mat'#[:]
    plotspiketrain(whole_mat)#|>display
    savefig(filename)

end
if !isfile("runinitial.jld")

    ##
    # ns0 is number of spikes per cell
    ##
    times, ns0, ustd_mean = runinitial(p,w0Index, w0Weights, nc0)
    save("runinitial.jld", "times", times, "ns0", ns0,"ustd",ustd_mean)

elseif isfile("runinitial.jld")
    @load "runinitial.jld"
end

#scatter(ns0,times)
#savefig("Spike_Rasterx.png")
#scatter(times,ns0)
#savefig("Spike_Rastery.png")

xs = []
ys = []
for ci=1:p.Ncells
    push!(xs, times[ci]*100)
    push!(ys, ci)
end
nt = isempty(x) ? 0 : maximum(y)
ms =4.45*size[2]/(nt+5)
#scatter(xs, ys)
Plots.scatter!(xs,ys;label="SpikeTrain",markershape=:vline,markersize=ms,markerstrokewidth = 0.2)

savefig("Better_Spike_Rastery.png")


xs2 = []
ys2 = []
for ci=1:p.Ncells

    push!(xs2,collect(findall(times[ci].>0)))
    @show(xs2)
    push!(ys2, ci)
end
scatter(xs2, ys2)
savefig("Better_Spike_Rastery1.png")

timess_cat = hcat(times)
#ps = Union{Plot,Context}[]
nneurons = p.Ncells
Nsteps = Int(p.train_time / p.dt) 
simtvec = [ dt*ti for ti=1:Nsteps ] 

psth = fit(Histogram, timess_cat, simtvec)
df = DataFrame(t=simtvec,
                model=psth.weights)
  

#append!(ps, fill(Compose.context(), nrows*ncols-nneurons))
savefig("Spike_Rasteryx.png")


# select cells to be trained
rtarg_lickright = load(dirALM * "movingrate_Pyr1Hz_lickright.jld", "Pyr")
rtarg_lickleft = load(dirALM * "movingrate_Pyr1Hz_lickleft.jld", "Pyr")

# sample a subset of neurons to be trained
Npyr = size(rtarg_lickleft)[1] # Npyr = 1824
Ntrained = Npyr # Num of trained neurons equals Npyr. More generally, Int(p.Ne * p.fracTrained)    
sampledNeurons = sort(shuffle(collect(1:Npyr))[1:Ntrained])

@show(length(sampledNeurons))
rtarg_lickright = rtarg_lickright[sampledNeurons,:]
rtarg_lickleft = rtarg_lickleft[sampledNeurons,:]
rtarg_mean = (rtarg_lickright + rtarg_lickleft)/2
almOrd, matchedCells = genCellsTrained(rtarg_mean, ns0)

wpWeightFfwd, wpWeightIn, wpWeightOut, wpIndexIn, wpIndexOut, wpIndexConvert, ncpIn, ncpOut = genPlasticWeights(p, nc0, ns0, matchedCells)

# select plastic weights to be trained
#if !isfile("plastic_weights.jld")
    #wpWeightFfwd, wpWeightIn, wpWeightOut, wpIndexIn, wpIndexOut, wpIndexConvert, ncpIn, ncpOut = genPlasticWeights(p, w0Index, nc0, ns0, matchedCells)

#    save("plastic_weights.jld", "wpWeightOut", wpWeightOut, "wpWeightIn", wpWeightIn)#,"ustd",ustd)
#end
postcells = filter(x->x!=0, wpWeightOut)

wpWeightOut = sparse(wpWeightOut)


stim = Vector{Array{Float64,2}}()
stim_R = genStim(p)
stim_L = genStim(p)
push!(stim, stim_R)
push!(stim, stim_L)

# load targets
xtarg = Vector{Array{Float64,2}}()
utarg_R = transpose(load(dirALM * "utarg1Hz_lickright.jld", "utarg"))[:,sampledNeurons]
utarg_L = transpose(load(dirALM * "utarg1Hz_lickleft.jld", "utarg"))[:,sampledNeurons]
push!(xtarg, utarg_R)
push!(xtarg, utarg_L)


# generate drive
ffwdRate_mean = 10.0
ffwdRate = Vector{Array{Float64,2}}()
ffwdRate_R = genffwdRate(p, ffwdRate_mean)
ffwdRate_L = genffwdRate(p, ffwdRate_mean)
push!(ffwdRate, ffwdRate_R)
push!(ffwdRate, ffwdRate_L)

#----------- save files --------------#
fname_param = dirNetwork * "p.jld"
fname_w0Index = dirNetwork * "w0Index.jld"
fname_w0Weights = dirNetwork * "w0Weights.jld"
fname_nc0 = dirNetwork * "nc0.jld"
fname_wpIndexIn = dirNetwork * "wpIndexIn.jld"
fname_wpIndexOut = dirNetwork * "wpIndexOut.jld"
fname_wpIndexConvert = dirNetwork * "wpIndexConvert.jld"
fname_ncpIn = dirNetwork * "ncpIn.jld"
fname_ncpOut = dirNetwork * "ncpOut.jld"
fname_stim_R = dirNetwork * "stim_R.jld"
fname_stim_L = dirNetwork * "stim_L.jld"
fname_ffwdRate = dirNetwork * "ffwdRate.jld"

#fname_uavg = dirNetwork * "uavg.jld"
fname_almOrd = dirNetwork * "almOrd.jld"
fname_matchedCells = dirNetwork * "matchedCells.jld"

save(fname_param,"p", p)
save(fname_w0Index,"w0Index", w0Index)
save(fname_w0Weights,"w0Weights", w0Weights)
save(fname_nc0,"nc0", nc0)
save(fname_wpIndexIn,"wpIndexIn", wpIndexIn)
save(fname_wpIndexOut,"wpIndexOut", wpIndexOut)
save(fname_wpIndexConvert,"wpIndexConvert", wpIndexConvert)
save(fname_ncpIn,"ncpIn", ncpIn)
save(fname_ncpOut,"ncpOut", ncpOut)
save(fname_stim_R,"stim", stim_R)
save(fname_stim_L,"stim", stim_L)
save(fname_ffwdRate,"ffwdRate", ffwdRate)

#save(fname_uavg,"uavg", uavg)
save(fname_almOrd,"almOrd", almOrd)
save(fname_matchedCells,"matchedCells", matchedCells)


if !isfile("existing_test.jld")

    #----------- run train --------------#
    wpWeightIn, wpWeightOut, wpWeightFfwd = runtrain(dirNetwork,p,w0Index,w0Weights,nc0, stim, xtarg,
    wpWeightFfwd, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut, ncpIn, ncpOut, 
    almOrd, matchedCells, ffwdRate)

    wpWeightOut = sparse(wpWeightOut)
    wpWeightIn = sparse(wpWeightIn)
    save("existing_test.jld", "wpWeightIn", wpWeightIn, "wpWeightOut", wpWeightOut,"wpWeightFfwd",wpWeightFfwd)
else
    @load existing_test.jld
    pcor_mean = runperformance(p,w0Index,w0Weights,nc0,wpIndexOut,wpWeightOut,ncpOut,stim,xtarg,almOrd,matchedCells,ffwdRate,wpWeightFfwd)
    (xtotal, xebal, xibal, xplastic, times, ns, vtotal_exccell, vtotal_inhcell, vebal_exccell, vibal_exccell, vebal_inhcell, vibal_inhcell, vplastic_exccell, vplastic_inhcell) = 
    runtest(p,w0Index,w0Weights,nc0,wpIndexOut,wpWeightOut,ncpOut,stim,ffwdRate,wpWeightFfwd)
    plot_spikes(ns,times,"Spike_Raster1.png")
end

##
# Number of spikes per cell
##
@show(unique(ns))
@show(length(times))


#----------- save files --------------#
fname_wpWeightIn = dirNetwork * "wpWeightIn.jld"
fname_wpWeightOut = dirNetwork * "wpWeightOut.jld"
fname_wpWeightFfwd = dirNetwork * "wpWeightFfwd.jld"

save(fname_wpWeightIn,"wpWeightIn", wpWeightIn)
save(fname_wpWeightOut,"wpWeightOut", wpWeightOut)
save(fname_wpWeightFfwd,"wpWeightFfwd", wpWeightFfwd)
