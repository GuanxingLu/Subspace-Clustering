function A = gnd2pair11(gnd, Omega)
    num_train = length(gnd);
    A = zeros(num_train);
    label_num = length(Omega);

    for m = 1:label_num-1
       for n = m+1:label_num
           if gnd(Omega(m)) == gnd(Omega(n))
               A(Omega(m),Omega(n)) = 1;
           elseif gnd(Omega(m)) ~= gnd(Omega(n))
               A(Omega(m),Omega(n)) = -1;
           end
       end
    end

    A = (A+A');
end