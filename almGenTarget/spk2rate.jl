function spk2rate(ntrial, spikeTime, eventTime, wid, timev, start_time)

    rates = zeros(length(timev))

    for t = 1:ntrial
        nspk = length(spikeTime[t])
        if nspk > 0
            spk = spikeTime[t] .- eventTime[t]
            for ii = 1:length(spk)
                if spk[ii] > start_time && spk[ii] < 0
                    kk = -Int(floor(spk[ii]/wid)) + 1
                    rates[kk] += 1 / ntrial / wid
                end
            end
        end
    end

    return rates

end