function runperformance(p,w0Index,w0Weights,nc0,wpIndexOut,wpWeightOut,ncpOut,stim,xtarg,almOrd,matchedCells,ffwdRate,wpWeightFfwd)

    xtotal, xebal, xibal, xplastic, times, ns, 
    vtotal_exc, vtotal_inh, vebal_exc, vibal_exc, 
    vebal_inh, vibal_inh, vplastic_exc, vplastic_inh = runtest(p,w0Index,w0Weights,nc0,wpIndexOut,wpWeightOut,ncpOut,stim,ffwdRate,wpWeightFfwd)

    tlen = size(xtotal)[1]
    pcor = zeros(length(almOrd))    
    for nid = 1:length(almOrd)
        ci_alm = almOrd[nid] # alm neuron
        ci = matchedCells[nid] # model neuron

        xtarg_slice = @view xtarg[1:tlen,ci_alm]
        xtotal_slice = @view xtotal[:,ci]
        pcor[nid] = cor(xtarg_slice, xtotal_slice)
    end

    pcor_mean = mean(pcor)

    return pcor_mean

end