function X21 = norm21(X)
    X_size = size(X);
    X21 = 0;
    for i = 1:X_size(1)
          X21 = X21 + norm(X(i,:));
    end
end