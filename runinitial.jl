using ProgressMeter


include("genStim.jl")


function runinitial(p,w0Index,w0Weights,nc0)
    stim_on = copy(p.stim_on)
    stim_off = copy(p.stim_off)
    train_time = copy(p.train_time)
    # copy param
    train_time = 20000
    # train_time = copy(p.train_time)    
    dt = copy(p.dt)
    Nsteps = Int(train_time / dt) # network param
    N#steps = copy(p.Nsteps) # network param
    Ncells = copy(p.Ncells)
    Ne = copy(p.Ne)
    Ni = copy(p.Ni)
    taue = copy(p.taue) # neuron param
    taui = copy(p.taui)
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
    ns = zeros(Int,Ncells) # Number of spikes incremented
    
    forwardInputsE = zeros(Ncells) #summed weight of incoming E spikes
    forwardInputsI = zeros(Ncells)
    forwardInputsEPrev = zeros(Ncells) #as above, for previous timestep
    forwardInputsIPrev = zeros(Ncells)
    
    xedecay = zeros(Ncells)
    xidecay = zeros(Ncells)
    
    v = threshe*rand(Ncells) #membrane voltage 
    
    lastSpike = -100.0*ones(Ncells) #time of last spike
    
    Nexam = 1000
    uavg = zeros(2*Nexam)
    utmp = zeros(Nsteps - Int(1000/p.dt),2*Nexam)
    t = 0.0
    r = zeros(Ncells)
    bias = zeros(Ncells)
    stim = genStim(p)

    @showprogress for ti=1:Nsteps#/500.0

        t = dt*ti;
        forwardInputsE .= 0.0;
        forwardInputsI .= 0.0;
        for ci = 1:Ncells
            xedecay[ci] += -dt*xedecay[ci]/tauedecay + forwardInputsEPrev[ci]/tauedecay
            xidecay[ci] += -dt*xidecay[ci]/tauidecay + forwardInputsIPrev[ci]/tauidecay
            synInput = xedecay[ci] + xidecay[ci]
    
            # excitatory (uavg)
            if ti > Int(1000/p.dt) && ci <= Nexam # 1000 ms
                uavg[ci] += (synInput + mu[ci]) / (Nsteps - Int(1000/p.dt)) # save synInput
            end
    
            # inhibitory (uavg)
            if ti > Int(1000/p.dt) && ci > Ne && ci <= Ne + Nexam # 1000 ms
                uavg[Nexam+ci-Ne] += (synInput + mu[ci]) / (Nsteps - Int(1000/p.dt)) # save synInput
            end
    
            # excitatory (ustd)
            if ti > Int(1000/p.dt) && ci <=Nexam
                utmp[ti - Int(1000/p.dt), ci] = synInput
            end
    
            # inhibitory (ustd)
            if ti > Int(1000/p.dt) && ci > Ne && ci <= Ne + Nexam # 1000 ms
                utmp[ti - Int(1000/p.dt), Nexam+ci-Ne] = synInput
            end
    
            #bias[ci] = mu[ci]
            if t > Int(stim_on) && t < Int(stim_off) 
                bias[ci] = mu[ci] + stim[ti-Int(stim_on/dt),ci]*20.0
            else
                bias[ci] = mu[ci]
            end
            #not in refractory period
            if t > (lastSpike[ci] + refrac)  
                v[ci] += dt*((1/tau[ci])*(bias[ci]-v[ci] + synInput))
                if v[ci] > thresh[ci]  #spike occurred
                    v[ci] = vre
                    lastSpike[ci] = t
                    ns[ci] = ns[ci]+1 # Number of spikes incremented
                    if ns[ci] <= maxTimes
                        times[ci,ns[ci]] = t
                    end


                    for j = 1:nc0[ci]
          
                        if w0Index[j,ci]!=0
                            if w0Weights[j,ci] > 0  #E synapse
                                forwardInputsE[w0Index[j,ci]] += w0Weights[j,ci]
                            elseif w0Weights[j,ci] < 0  #I synapse
                                forwardInputsI[w0Index[j,ci]] += w0Weights[j,ci]

                            end
                        end
                    end #end loop over synaptic projections
                end #end if(spike occurred)
                #@show(times[ci,ns[ci]])
            end #end not in refractory period
        end #end loop over neurons
        forwardInputsEPrev = copy(forwardInputsE)
        forwardInputsIPrev = copy(forwardInputsI)
    end #end loop over time
    print("\r")
    println("mean excitatory firing rate: ",mean(1000*ns[1:Ne]/train_time)," Hz")
    println("mean inhibitory firing rate: ",mean(1000*ns[(Ne+1):Ncells]/train_time)," Hz")
    
    ustd = std(utmp, dims=1)[:]
    ustd_mean = mean(ustd)
    
    return times, ns, ustd_mean
    
end
    