function [ rotm ] = quat2rotm( quat )
%% =========== PBA's Implementation ============== %%
qq = sqrt(quat(1)*quat(1)+quat(2)*quat(2)+quat(3)*quat(3)+quat(4)*quat(4));
qw = 0; qx = 0; qy = 0; qz = 0;
if (qq > 0)
    qw = quat(1) / qq;
    qx = quat(2) / qq;
    qy = quat(3) / qq;
    qz = quat(4) / qq;
else
    qw = 1;
    qx = 0; qy = 0; qz = 0;
end
rotm = zeros(3,3);
rotm(1,1) = qw * qw + qx * qx - qz * qz - qy * qy;
rotm(1,2) = 2 * qx * qy - 2 * qz * qw;
rotm(1,3) = 2 * qy * qw + 2 * qz * qx;
rotm(2,1) = 2 * qx * qy + 2 * qw * qz;
rotm(2,2) = qy * qy + qw * qw - qz * qz - qx * qx;
rotm(2,3) = 2 * qz * qy - 2 * qx * qw;
rotm(3,1) = 2 * qx * qz - 2 * qy * qw;
rotm(3,2) = 2 * qy * qz + 2 * qw * qx;
rotm(3,3) = qz * qz + qw * qw - qy * qy - qx * qx;
end

