function [K, R, C] = DecomposeCameraMatrix(P);

% P = K[R -RC]

ncam = size(P, 1)/3;
K= zeros(3*ncam, 3);
R= zeros(3*ncam, 3);
C= zeros(ncam, 3);

id = 0;
for i = 1:ncam
    r_norm = norm(P(i*3,1:3));
    
    if r_norm > 0
        M = P((i-1)*3+1:3*i,1:3)/r_norm;
        p = P((i-1)*3+1:3*i, 4)/r_norm;

        R0(3, :) = M(3, :);

        u = M(1, :)*M(3, :)';
        v = M(2, :)*M(3, :)';

        fy = norm(M(2, :) - v * M(3, :));
        R0(2, :) = (M(2, :) - v*M(3, :))/fy;

        s = M(1, :)*R0(2, :)';
        fx = norm(M(1, :) - u*R0(3, :)- s*R0(2, :));
        R0(1, :) = (M(1, :) - s*R0(2, :) - u*R0(3, :))/fx;

        K0 = [fx s u
              0 fy v
               0 0 1];
        C0 = -inv(K0*R0)*p;
        R(id*3+1:(id+1)*3, :) = R0;
        K(id*3+1:(id+1)*3, :) = K0;
        C(id+1, :) = C0';
        id = id+1;
    end;
end;

K = K(1:id*3, :);
R = R(1:id*3, :);
C = C(1:id, :);

