using Distributions
using PyCall
using PyPlot
using LinearAlgebra
using Random
using JLD
using NLsolve

include("ricciardi.jl")
include("solveRicci.jl")
include("param.jl")
include("genWeights.jl")
include("runinitial2.jl")

dirtarget_selectivity = "/data/kimchm/data/dale/janelia/s1alm/target/selectivity/"
dirutarg_pyr_lickright = "/data/kimchm/data/dale/janelia/s1alm/target/selectivity/utarg/pyr/lickright/"
dirutarg_pyr_lickleft = "/data/kimchm/data/dale/janelia/s1alm/target/selectivity/utarg/pyr/lickleft/"


#---------- mean rate to be learned ----------#
rtarg_lickright = load(dirtarget_selectivity * "movingrate_Pyr_lickright.jld", "Pyr")
rtarg_lickleft = load(dirtarget_selectivity * "movingrate_Pyr_lickleft.jld", "Pyr")

#---------- target function for synaptic inputs ----------#
utarg_lickright = zeros(size(rtarg_lickright))
utarg_lickleft = zeros(size(rtarg_lickleft))
Npyr = size(utarg_lickleft)[1]
for ci = 1:Npyr
        utarg_lickright[ci,:] = load(dirutarg_pyr_lickright * "utarg$(ci).jld", "utarg")
        utarg_lickleft[ci,:] = load(dirutarg_pyr_lickleft * "utarg$(ci).jld", "utarg")
end


#---------- remove neurons with rate < 1Hz ----------#
meanRate_lickright = mean(rtarg_lickright, dims=2)[:]
meanRate_lickleft = mean(rtarg_lickleft, dims=2)[:]
idx1Hz_lickright = meanRate_lickright .> 1.0
idx1Hz_lickleft = meanRate_lickleft .> 1.0
idx1Hz = idx1Hz_lickright .* idx1Hz_lickleft

rtarg_lickright = rtarg_lickright[idx1Hz, :]
rtarg_lickleft = rtarg_lickleft[idx1Hz, :]
utarg_lickright = utarg_lickright[idx1Hz, :]
utarg_lickleft = utarg_lickleft[idx1Hz, :]

save(dirtarget_selectivity * "movingrate_Pyr1Hz_lickright.jld", "Pyr", rtarg_lickright)
save(dirtarget_selectivity * "movingrate_Pyr1Hz_lickleft.jld", "Pyr", rtarg_lickleft)

save(dirutarg_pyr_lickright * "utarg1Hz.jld", "utarg", utarg_lickright)
save(dirutarg_pyr_lickleft * "utarg1Hz.jld", "utarg", utarg_lickleft)
