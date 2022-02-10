function alpha1 = computeLambda_mat(X)
[~,n] = size(X);
B = zeros(n);
for i = 1:n
    for j = 1:n
        if j~=i
            B(i,j) = norm(X(:,i)'*X(:,j),1);
        end
    end
end
alpha1 = min(max(B,[],2));
end
