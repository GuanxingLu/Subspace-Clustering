function spectralNorm = lansvd(B)

[~,S,~] = svd(B'*B,'econ');
spectralNorm = sqrt(max(S));

end