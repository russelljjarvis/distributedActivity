using Distributions
using PyCall
using PyPlot
using DelimitedFiles
using LinearAlgebra
using Random
using SparseArrays
using JLD
using MultivariateStats

# modifiable parameters
g_list = [1.0, 1.5]
ii = 1 # g = 1.0

# lam_list = collect(1.0:1.0:3.0) # 0.2
# lam_list = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
lam_list = [0.04, 0.08, 0.2, 0.6, 1.0]
jj = 1 

mu_list = [0.0, 2.0, 4.0, 8.0]
kk = 4 # mu = 8.0

L_list = [2.0, 4.0, 6.0, 8.0, 10.0]
# ll = parse(Int64,ARGS[1]) # L = 2.0
ll = 3

wpscale_list = [1.0, 2.0, 3.0]
mm = 2 # wpscale = 2.0. To improve dale's law try wpscale = 1.0 (Jan 6)

taup_list = [25.0, 50.0, 75.0, 100.0, 150.0]
# nn = parse(Int64,ARGS[2])
nn = 5


# include("explainedVar.jl")
# include("sharedVar.jl")

# include("figure/plt_sharedvar_pyrexc.jl")
# include("figure/plt_sharedvar_fsinh.jl")
# # include("figure/plt_sharedvar_data_ei.jl")
# include("figure/plt_pca.jl")
# include("figure/plt_cormat.jl")

temporalAvg = "noavg" # "avg", "noavg"

dirsim = "/data/kimchm/data/dale/janelia/trained/basic/sim/" * temporalAvg * "/"
dirsim_bal = "/data/kimchm/data/dale/janelia/trained/basic/simffwd/balanced/" * temporalAvg * "/"
dirtarget_selectivity = "/data/kimchm/data/dale/janelia/s1alm/target/selectivity/"
dirutargPyr = "/data/kimchm/data/dale/janelia/s1alm/target/utarg/"
dirutargFS = "/data/kimchm/data/dale/janelia/s1alm/target/utargFS/"
dirfig_selectivity = "/data/kimchm/data/dale/janelia/figure/selectivity/"

if ~ispath(dirfig_selectivity)
    mkpath(dirfig_selectivity)
end

#-------- load Pyr and FS --------#
pyr_right = transpose(load(dirtarget_selectivity * "movingrate_Pyr_lickright.jld", "Pyr"))[:,:]
fs_right = transpose(load(dirtarget_selectivity * "movingrate_FS_lickright.jld", "FS"))[:,:]

pyr_left = transpose(load(dirtarget_selectivity * "movingrate_Pyr_lickleft.jld", "Pyr"))[:,:]
fs_left = transpose(load(dirtarget_selectivity * "movingrate_FS_lickleft.jld", "FS"))[:,:]


#-------- plot diff Pyr --------#
pyr_right_norm = zeros(size(pyr_right))
pyr_left_norm = zeros(size(pyr_left))
for ci = 1:size(pyr_right)[2]
    avgrate = (mean(pyr_left[:,ci]) + mean(pyr_right[:,ci]))/2
    avgstd = (std(pyr_left[:,ci]) + std(pyr_right[:,ci]))/2
    pyr_right_norm[:,ci] = (pyr_right[:,ci] .- avgrate)/avgstd
    pyr_left_norm[:,ci] = (pyr_left[:,ci] .- avgrate)/avgstd
end

diffPyr = pyr_left - pyr_right
diffPyrmean = mean(diffPyr, dims=1)[:]
sortPyr = sortperm(diffPyrmean)
diffPyrsorted = diffPyr[:,sortPyr]

diffPyr_norm = pyr_left_norm - pyr_right_norm
diffPyrmean_norm = mean(diffPyr_norm, dims=1)[:]
sortPyr_norm = sortperm(diffPyrmean_norm)
diffPyrsorted_norm = diffPyr_norm[:,sortPyr_norm]

figure(figsize=(3.7,3))
imshow(transpose(diffPyrsorted)[:,:], cmap="bwr", vmin=-10, vmax=10, interpolation="None", aspect="auto")
colorbar()
tight_layout()

savefig(dirfig_selectivity * "diffPyr.png", dpi=300)


figure(figsize=(3.7,3))
imshow(transpose(diffPyrsorted_norm)[:,:], cmap="bwr", vmin=-5, vmax=5, interpolation="None", aspect="auto")
colorbar()
tight_layout()

savefig(dirfig_selectivity * "diffPyr_norm.png", dpi=300)


#-------- plot diff FS --------#
fs_right_norm = zeros(size(fs_right))
fs_left_norm = zeros(size(fs_left))
for ci = 1:size(fs_right)[2]
    avgrate = (mean(fs_left[:,ci]) + mean(fs_right[:,ci]))/2
    avgstd = (std(fs_left[:,ci]) + std(fs_right[:,ci]))/2
    fs_left_norm[:,ci] = (fs_left[:,ci] .- avgrate)/avgstd
    fs_right_norm[:,ci] = (fs_right[:,ci] .- avgrate)/avgstd
end

diffFS = fs_left - fs_right
diffFSmean = mean(diffFS, dims=1)[:]
sortFS = sortperm(diffFSmean)
diffFSsorted = diffFS[:,sortFS]

diffFS_norm = fs_left_norm - fs_right_norm
diffFSmean_norm = mean(diffFS_norm, dims=1)[:]
sortFS_norm = sortperm(diffFSmean_norm)
diffFSsorted_norm = diffFS_norm[:,sortFS_norm]

figure(figsize=(3.7,3))
imshow(transpose(diffFSsorted)[:,:], cmap="bwr", vmin=-10, vmax=10, interpolation="None", aspect="auto")
colorbar()
tight_layout()

savefig(dirfig_selectivity * "diffFS.png", dpi=300)



figure(figsize=(3.7,3))
imshow(transpose(diffFSsorted_norm)[:,:], cmap="bwr", vmin=-5, vmax=5, interpolation="None", aspect="auto")
colorbar()
tight_layout()

savefig(dirfig_selectivity * "diffFS_norm.png", dpi=300)




figure(figsize=(7,6))
subplot(221)
plot(diffPyrmean[sortPyr], color="limegreen", marker="o", ms=4, linestyle="", label="Pyr")
plot(collect(1:length(sortPyr)), zeros(length(sortPyr)), color="gray", linestyle="--") 
xlabel("Pyr neuron", fontsize=12)
ylabel(L"$\Delta$" * "rate", fontsize=12)
xticks(fontsize=12)
yticks(fontsize=12)
legend(frameon=false)

subplot(222)
plot(diffFSmean[sortFS], color="darkorange", marker="o", ms=4, linestyle="", label="FS")
plot(collect(1:length(sortFS)), zeros(length(sortFS)), color="gray", linestyle="--")
xlabel("FS neuron", fontsize=12)
ylabel(L"$\Delta$" * "rate", fontsize=12)
xticks(fontsize=12)
yticks(fontsize=12)
legend(frameon=false)

subplot(223)
plot(diffPyrmean_norm[sortPyr_norm], color="limegreen", marker="o", ms=4, linestyle="", label="Pyr norm")
plot(collect(1:length(sortPyr)), zeros(length(sortPyr)), color="gray", linestyle="--")
xlabel("Pyr neuron", fontsize=12)
ylabel(L"$\Delta$" * "rate (norm.)", fontsize=12)
xticks(fontsize=12)
yticks(fontsize=12)
legend(frameon=false)

subplot(224)
plot(diffFSmean_norm[sortFS_norm], color="darkorange", marker="o", ms=4, linestyle="", label="FS norm")
plot(collect(1:length(sortFS)), zeros(length(sortFS)), color="gray", linestyle="--")
xlabel("FS neuron", fontsize=12)
ylabel(L"$\Delta$" * "rate (norm.)", fontsize=12)
xticks(fontsize=12)
yticks(fontsize=12)
legend(frameon=false)

tight_layout()

savefig(dirfig_selectivity * "selectivity.png", dpi=300)
