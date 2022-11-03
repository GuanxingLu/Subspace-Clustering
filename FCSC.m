function W = FCSC(Z , B)
%Fully Constrained Spectral Clustering
[m,n]=size(B);
W = zeros(m,n);
for i = 1:m
    for j = 1:n
        if B(i, j)>=0
            W(i,j) = 1 - (1-B(i,j))*(1 - Z(i,j));   %若B是1，则该项是1
        else
            W(i,j) = (1+ B(i, j))*Z(i,j);   %若B是-1，则该项是0
        end
    end
end
    
end