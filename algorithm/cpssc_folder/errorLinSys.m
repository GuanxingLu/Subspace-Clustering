function err = errorLinSys(X, J)
chg = abs(X-X*J);
err = max(chg(:));
end