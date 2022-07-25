function funMovAvg(x,wid)

Nsteps = size(x)[2]
movavg = zeros(size(x))
for i = 1:Nsteps
    Lind = maximum([i-wid, 1])
    Rind = minimum([i+wid, Nsteps])
    xslice = @view x[:,Lind:Rind]
    movavg[:,i] = mean(xslice, dims=2)[:]
end
    
return movavg

end