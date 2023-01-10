using Distributions
using ProgressMeter

function runtest(p,w0Index,w0Weights,nc0,wpIndexOut,wpWeightOut,ncpOut,stim,ffwdRate,wpWeightFfwd)

# copy param
nloop = copy(p.nloop) # train param
penlambda = copy(p.penlambda)
penmu = copy(p.penmu)
fracTrained = copy(p.fracTrained)
learn_every = copy(p.learn_every)
stim_on = floor(Int,copy(p.stim_on)/10.0)
stim_off = floor(Int,copy(p.stim_off)/10.0)
train_time = floor(Int,copy(p.train_time)/10.0)
dt = copy(p.dt) # time param
Nsteps = floor(Int,copy(p.Nsteps)/10.0) 
Ncells = copy(p.Ncells) # network param
Ne = copy(p.Ne)
Ni = copy(p.Ni)
taue = copy(p.taue) # neuron param
taui = copy(p.taui)
sqrtK = copy(p.sqrtK)
threshe = copy(p.threshe)
threshi = copy(p.threshi)
refrac = copy(p.refrac)
vre = copy(p.vre)
muemin = copy(p.muemin) # external input
muemax = copy(p.muemax)
muimin = copy(p.muimin)
muimax = copy(p.muimax)
tauedecay = copy(p.tauedecay) # synaptic time
tauidecay = copy(p.tauidecay)
taudecay_plastic = copy(p.taudecay_plastic)
maxrate = copy(p.maxrate)

# set up variables
mu = zeros(Ncells)
mu[1:Ne] = (muemax-muemin)*rand(Ne) .+ muemin
mu[(Ne+1):Ncells] = (muimax-muimin)*rand(Ni) .+ muimin

thresh = zeros(Ncells)
thresh[1:Ne] .= threshe
thresh[(1+Ne):Ncells] .= threshi

tau = zeros(Ncells)
tau[1:Ne] .= taue
tau[(1+Ne):Ncells] .= taui

maxTimes = round(Int,maxrate*train_time/1000)
times = zeros(Ncells,maxTimes)
ns = zeros(Int,Ncells)
times_ffwd = zeros(p.Lffwd, maxTimes)
ns_ffwd = zeros(Int, p.Lffwd)

forwardInputsE = zeros(Ncells) #summed weight of incoming E spikes
forwardInputsI = zeros(Ncells)
forwardInputsP = zeros(Ncells)
forwardInputsEPrev = zeros(Ncells) #as above, for previous timestep
forwardInputsIPrev = zeros(Ncells)
forwardInputsPPrev = zeros(Ncells)
forwardSpike = zeros(Ncells)
forwardSpikePrev = zeros(Ncells)

xedecay = zeros(Ncells)
xidecay = zeros(Ncells)
xpdecay = zeros(Ncells)
synInputBalanced = zeros(Ncells)

v = zeros(Ncells) #membrane voltage 

lastSpike = -100.0*ones(Ncells) #time of last spike
  
t = 0.0
bias = zeros(Ncells)

learn_nsteps = Int((p.train_time - p.stim_off)/p.learn_every)
learn_seq = 1
example_neurons = 25
wid = 50
widInc = Int(2*wid/p.learn_every - 1)

vtotal_exccell = zeros(Nsteps,example_neurons)
vtotal_inhcell = zeros(Nsteps,example_neurons)
vebal_exccell = zeros(Nsteps,example_neurons)
vibal_exccell = zeros(Nsteps,example_neurons)
vebal_inhcell = zeros(Nsteps,example_neurons)
vibal_inhcell = zeros(Nsteps,example_neurons)
vplastic_exccell = zeros(Nsteps,example_neurons)
vplastic_inhcell = zeros(Nsteps,example_neurons)
xtotal = zeros(learn_nsteps,Ncells)
xebal = zeros(learn_nsteps,Ncells)
xibal = zeros(learn_nsteps,Ncells)
xplastic = zeros(learn_nsteps,Ncells)
xtotalcnt = zeros(learn_nsteps)
xebalcnt = zeros(learn_nsteps)
xibalcnt = zeros(learn_nsteps)
xplasticcnt = zeros(learn_nsteps)


#
# Monday next week.
#
@showprogress for ti=1:Nsteps
    if mod(ti,Nsteps/100) == 1  #print percent complete
        print("\r",round(Int,100*ti/Nsteps))
    end

    t = dt*ti;
    forwardInputsE .= 0.0;
    forwardInputsI .= 0.0;
    forwardInputsP .= 0.0;
    forwardSpike .= 0.0;
    rndFfwd = rand(p.Lffwd)

    for ci = 1:Ncells
        xedecay[ci] += -dt*xedecay[ci]/tauedecay + forwardInputsEPrev[ci]/tauedecay
        xidecay[ci] += -dt*xidecay[ci]/tauidecay + forwardInputsIPrev[ci]/tauidecay
        xpdecay[ci] += -dt*xpdecay[ci]/taudecay_plastic + forwardInputsPPrev[ci]/taudecay_plastic
        synInputBalanced[ci] = xedecay[ci] + xidecay[ci]
        synInput = synInputBalanced[ci] + xpdecay[ci]

        # # saved for visualization
        # if ci <= example_neurons
        #     vtotal_exccell[ti,ci] = synInput
        #     vebal_exccell[ti,ci] = xedecay[ci]
        #     vibal_exccell[ti,ci] = xidecay[ci]
        #     vplastic_exccell[ti,ci] = xpdecay[ci]
        # elseif ci >= Ncells - example_neurons + 1
        #     vtotal_inhcell[ti,ci-Ncells+example_neurons] = synInput
        #     vebal_inhcell[ti,ci-Ncells+example_neurons] = xedecay[ci]
        #     vibal_inhcell[ti,ci-Ncells+example_neurons] = xidecay[ci]
        #     vplastic_inhcell[ti,ci-Ncells+example_neurons] = xpdecay[ci]
        # end

        # save rolling average for analysis
        if t > Int(stim_off) && t <= Int(train_time) && mod(t,1.0) == 0
            xtotal[:,ci], xtotalcnt = funRollingAvg(p,t,wid,widInc,learn_nsteps,xtotal[:,ci],xtotalcnt,synInput,ci)
        end
        #@show(bias)
        #@show(ci)        
        # external input
        if t > Int(stim_on) && t < Int(stim_off)
            bias[ci] = mu[ci] + stim[1][ti-Int(stim_on/dt),ci]

            #@show(stim[ti-Int(stim_on/dt),ci])
            bias[ci] = mu[ci] + stim[2][ti-Int(stim_on/dt),ci]
        else
            bias[ci] = mu[ci]
        end
        #@show(ci)
        #@show(size(stim[1]))
        #@show(size(stim[2]))

        #not in refractory period
        if t > (lastSpike[ci] + refrac)  
            v[ci] += dt*((1/tau[ci])*(bias[ci]-v[ci] + synInput))
            if v[ci] > thresh[ci]  #spike occurred
                v[ci] = vre
                forwardSpike[ci] = 1.
                lastSpike[ci] = t
                ns[ci] = ns[ci]+1
                if ns[ci] <= maxTimes
                    times[ci,ns[ci]] = t
                end
                for j = 1:nc0[ci]
                    wgt = w0Weights[j,ci]
                    cell = w0Index[j,ci]
                    if wgt > 0  #E synapse
                        forwardInputsE[cell] += wgt
                    elseif wgt < 0  #I synapse
                        forwardInputsI[cell] += wgt
                    end
                end #end loop over synaptic projections
                for j = 1:ncpOut[ci]
                    cell = Int(wpIndexOut[j,ci])
                    forwardInputsP[cell] += wpWeightOut[j,ci]
                end
            end #end if(spike occurred)
        end #end not in refractory period
    end #end loop over neurons

    # External input to trained excitatory neurons

    #@show(ffwdRate)
    if ti > Int(stim_off/dt)
        tidx = ti - Int(stim_off/dt)
        for ci = 1:p.Lffwd
            # # if training, filter the spikes
            #s[ci] += -dt*s[ci]/taudecay_plastic + ffwdSpikePrev[ci]/taudecay_plastic

            # if Poisson neuron spiked
            #@show(ffwdRate[1][tidx,ci])
            #@show(rndFfwd[ci])

            if rndFfwd[ci] < ffwdRate[1][tidx,ci]/(1000/p.dt)
                #fwdSpike[ci] = 1.
                ns_ffwd[ci] = ns_ffwd[ci]+1
                if ns_ffwd[ci] <= maxTimes
                    times_ffwd[ci,ns_ffwd[ci]] = t
                end
                #@show(wpWeightFfwd)
                #@show(size(wpWeightFfwd))
                
                #for j = 1:Ne
                #    forwardInputsP[j] += wpWeightFfwd[j,ci]
                #end #end loop over synaptic projections
            end #end if spiked
        end #end loop over ffwd neurons
    end #end ffwd input
    

    forwardInputsEPrev = copy(forwardInputsE)
    forwardInputsIPrev = copy(forwardInputsI)
    forwardInputsPPrev = copy(forwardInputsP)
    forwardSpikePrev = copy(forwardSpike) # if training, compute spike trains

end #end loop over time
print("\r")

for k = 1:learn_nsteps
    xtotal[k,:] = xtotal[k,:]/xtotalcnt[k]
    # xebal[k,:] = xebal[k,:]/xebalcnt[k]
    # xibal[k,:] = xibal[k,:]/xibalcnt[k]
    # xplastic[k,:] = xplastic[k,:]/xplasticcnt[k]
end

return xtotal, xebal, xibal, xplastic, times, ns, vtotal_exccell, vtotal_inhcell, vebal_exccell, vibal_exccell, vebal_inhcell, vibal_inhcell, vplastic_exccell, vplastic_inhcell


end
