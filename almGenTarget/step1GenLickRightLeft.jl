using Distributions
using PyCall
using PyPlot
using DelimitedFiles
using LinearAlgebra
using Random
using SparseArrays
using JLD
using MAT

include("spk2rate.jl")
include("funMovAvg.jl")
include("genRates.jl")

# pre-sample: 1.2 sec
# sample: 1 sec
# delay: 2 sec
# go cue: 0.1 sec
# response: 1.5 sec

# optical stimulus: starts -2.5s from go, duration 0.4s

# julia> keys(alm)
#   "early_lick":           no early 
#   "trial"                 --
#   "outcome":              hit, miss
#   "spike_times"           --
#   "trial_event_time":     go time
#   "trial_uid"             --
#   "trial_event_type":     go 
#   "session"               --
#   "task":                 s1 stim 
#   "hemisphere"            --
#   "unit_dv_location"      --
#   "cell_type":            Pyr, FS, not classified 
#   "subject_id"            --
#   "trial_instruction":    left, right
#   "unit_uid"              --
#   "trial_type_name":      2984 elements
#   "unit"                  --
#   "brain_area":           ALM

start_time = -2.0 #sec from go cue

dirdata = "/data/kimchm/data/dale/janelia/s1alm/"
dirtarget_selectivity = "/data/kimchm/data/dale/janelia/s1alm/target/selectivity/"
# dirfig_selectivity = "/data/kimchm/data/dale/janelia/figure/selectivity/"

if ~ispath(dirtarget_selectivity)
    mkpath(dirtarget_selectivity)
end

fid = matopen(dirdata * "data_nodistractor_trials_ALM.mat")
alm = read(fid)["data"]
close(fid)

#----- photo-stimulated trials -----#
#----- i.e., trial_instruction = "left" -----#
cell_type_Pyr = "Pyr"
cell_type_FS = "FS"
trial_instruction = "left"
outcome = "hit"

idx_cell_type_Pyr = alm["cell_type"] .== cell_type_Pyr
idx_cell_type_FS = alm["cell_type"] .== cell_type_FS
idx_trial_instruction = alm["trial_instruction"] .== trial_instruction
idx_outcome = alm["outcome"] .== outcome
idx_Pyr = idx_cell_type_Pyr .* idx_trial_instruction .* idx_outcome
idx_FS = idx_cell_type_FS .* idx_trial_instruction .* idx_outcome

println("lick left - Pyr")
Lrates_Pyr, Lmovingrate_Pyr, Lcells_Pyr, Lunits_Pyr, LspikeTime_Pyr, LeventTime_Pyr = genRates(alm, idx_Pyr, start_time)
println("lick left - FS")
Lrates_FS, Lmovingrate_FS, Lcells_FS, Lunits_FS, LspikeTime_FS, LeventTime_FS = genRates(alm, idx_FS, start_time)


#----- i.e., trial_instruction = "right" -----#
cell_type_Pyr = "Pyr"
cell_type_FS = "FS"
trial_instruction = "right"
outcome = "hit"

idx_cell_type_Pyr = alm["cell_type"] .== cell_type_Pyr
idx_cell_type_FS = alm["cell_type"] .== cell_type_FS
idx_trial_instruction = alm["trial_instruction"] .== trial_instruction
idx_outcome = alm["outcome"] .== outcome
idx_Pyr = idx_cell_type_Pyr .* idx_trial_instruction .* idx_outcome
idx_FS = idx_cell_type_FS .* idx_trial_instruction .* idx_outcome

println("lick right - Pyr")
Rrates_Pyr, Rmovingrate_Pyr, Rcells_Pyr, Runits_Pyr, RspikeTime_Pyr, ReventTime_Pyr = genRates(alm, idx_Pyr, start_time)
println("lick right - FS")
Rrates_FS, Rmovingrate_FS, Rcells_FS, Runits_FS, RspikeTime_FS, ReventTime_FS = genRates(alm, idx_FS, start_time)


#----- select FS cells present in both Left/Right -----#
Nfs_right = size(Rcells_FS)
Nfs_left = size(Lcells_FS)
fs_right_idx = falses(Nfs_right)
fs_left_idx = falses(Nfs_left)
for (ci_left, cell_left) in enumerate(Lcells_FS)
    ci_right = findall(x->x==cell_left, Rcells_FS)
    if !isempty(ci_right)
        fs_right_idx[ci_right[1]] = true
        fs_left_idx[ci_left] = true
    end
end

Rmovingrate_FS = Rmovingrate_FS[fs_right_idx, :]
Lmovingrate_FS = Lmovingrate_FS[fs_left_idx, :]
Rrates_FS = Rrates_FS[fs_right_idx, :]
Lrates_FS = Lrates_FS[fs_left_idx, :]


#----- select Pyr cells present in both Left/Right -----#
Npyr_right = size(Rcells_Pyr)
Npyr_left = size(Lcells_Pyr)
pyr_right_idx = falses(Npyr_right)
pyr_left_idx = falses(Npyr_left)
for (ci_left, cell_left) in enumerate(Lcells_Pyr)
    ci_right = findall(x->x==cell_left, Rcells_Pyr)
    if !isempty(ci_right)
        pyr_right_idx[ci_right[1]] = true
        pyr_left_idx[ci_left] = true
    end
end

Rmovingrate_Pyr = Rmovingrate_Pyr[pyr_right_idx, :]
Lmovingrate_Pyr = Lmovingrate_Pyr[pyr_left_idx, :]
Rrates_Pyr = Rrates_Pyr[pyr_right_idx, :]
Lrates_Pyr = Lrates_Pyr[pyr_left_idx, :]

Rcells_Pyr = Rcells_Pyr[pyr_right_idx]
Lcells_Pyr = Lcells_Pyr[pyr_left_idx]

save(dirtarget_selectivity * "movingrate_FS_lickright.jld", "FS", Rmovingrate_FS)
save(dirtarget_selectivity * "movingrate_FS_lickleft.jld", "FS", Lmovingrate_FS)
save(dirtarget_selectivity * "movingrate_Pyr_lickright.jld", "Pyr", Rmovingrate_Pyr)
save(dirtarget_selectivity * "movingrate_Pyr_lickleft.jld", "Pyr", Lmovingrate_Pyr)


save(dirtarget_selectivity * "rates_FS_lickright.jld", "FS", Rrates_FS)
save(dirtarget_selectivity * "rates_FS_lickleft.jld", "FS", Lrates_FS)
save(dirtarget_selectivity * "rates_Pyr_lickright.jld", "Pyr", Rrates_Pyr)
save(dirtarget_selectivity * "rates_Pyr_lickleft.jld", "Pyr", Lrates_Pyr)
