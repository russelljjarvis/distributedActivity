function genRates(alm, idx, start_time)

    units = alm["unit_uid"][idx]
    spikeTime = alm["spike_times"][idx]
    eventTime = alm["trial_event_time"][idx]
    
    cells = Int.(unique(units))
    Ncells = length(unique(units))
    
    wid = 0.02 # 20 ms
    timev = collect(start_time:wid:0)
    rates = zeros(Ncells, length(timev))
    num = 0
    for (cnt, ci) in enumerate(cells)
        num += 1

        ind = findall(x->x==ci, units)
        ntrial_ci = size(ind)[1]
        spikeTime_ci = spikeTime[ind]
        eventTime_ci = eventTime[ind]
        rates[cnt,:] = spk2rate(ntrial_ci, spikeTime_ci, eventTime_ci, wid, timev, start_time)
    end
    rates = reverse(rates, dims=2)
    movingrate = funMovAvg(rates,7)

    println("num cells: ", num)
    
    return rates, movingrate, cells, units, spikeTime, eventTime
end