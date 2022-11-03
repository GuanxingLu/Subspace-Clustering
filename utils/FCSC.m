function W = FCSC(Z , B)
%Fully Constrained Spectral Clustering
[m,n]=size(B);
W = zeros(m,n);
for i = 1:m
    for j = 1:n
        if B(i, j)>=0
            W(i,j) = 1 - (1-B(i,j))*(1 - Z(i,j));
        else
            W(i,j) = (1+ B(i, j))*Z(i,j);
        end
    end
end
    
end