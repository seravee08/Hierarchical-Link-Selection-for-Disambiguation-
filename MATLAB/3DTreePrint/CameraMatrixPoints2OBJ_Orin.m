function [] = CameraMatrixPoints2OBJ(P, X);

ncam = size(P, 1)/3;
npt = size(X, 1);
[K, R, C] = DecomposeCameraMatrix(P);

%check rotation matrix
if det(R(1:3, 1:3)) < 0
    %flip Y axis
    C(:, 2) = -C(:, 2);
    R(:, 2) = -R(:, 2);
    
    for i = 1:npt
        X(i, 2) = -X(i, 2);
    end;
end;

%compute camera point
%set camera focal length as the average distance between neibouring camera centers.
ncam = size(C, 1);
avg_dist = 0;
for i = 1:ncam
    min_dist = 0;
    for j = i+1:ncam
       dist = norm(C(i, :)-C(j,:));
       if min_dist == 0 | min_dist > dist
           min_dist = dist;
       end;
    end;
    if min_dist > 100
        continue;
    end;
    avg_dist = avg_dist + min_dist;
end;
f = 0.05;%avg_dist/ncam/2;

a = 2.0/3*f; 
b = 0.5*f;

a = a/2;
b = b/2;
c_campts = [0 0 0
            -a -b f
            a -b f
            a b f
            -a b f];
campts = zeros(5, 3, ncam);

for i = 1:ncam
    Rt = R((i-1)*3+1:i*3, :)';
    p0 = C(i, :)';
    p1 = Rt*c_campts(2, :)' + p0; 
    p2 = Rt*c_campts(3, :)' + p0; 
    p3 = Rt*c_campts(4, :)' + p0; 
    p4 = Rt*c_campts(5, :)' + p0;
    
    campts(:, :, i) = [p0';p1';p2';p3';p4'];
end;

% save camera as OBJ
mkdir('prepare');
objfn = 'prepare\cam.obj';
fid = fopen(objfn,'w');

if fid > 0
    for i = 1:ncam
        fprintf(fid, 'v %f %f %f\n', campts(1, 1, i), campts(1, 2, i), campts(1, 3, i));
        fprintf(fid, 'v %f %f %f\n', campts(2, 1, i), campts(2, 2, i), campts(2, 3, i));
        fprintf(fid, 'v %f %f %f\n', campts(3, 1, i), campts(3, 2, i), campts(3, 3, i));
        fprintf(fid, 'v %f %f %f\n', campts(4, 1, i), campts(4, 2, i), campts(4, 3, i));
        fprintf(fid, 'v %f %f %f\n', campts(5, 1, i), campts(5, 2, i), campts(5, 3, i));
    end;
    
    for i = 1:ncam
        for j = 1:5
            p(j) = j + (i-1)*5;
        end;
        
        for j = 1:4
            fprintf(fid, 'f %d %d %d\n', p(1), p(rem(j-1, 4) + 2), p(rem(j, 4)+2));
        end;
        fprintf(fid, 'f %d %d %d\n', p(2), p(4), p(3));
        fprintf(fid, 'f %d %d %d\n', p(2), p(5), p(4));
    end;
    
    fclose(fid);
end;

% save points as PLY
plyfn = 'prepare\pts.ply';
fid = fopen(plyfn, 'w');

if fid > 0
    fprintf(fid, 'ply\n');
    fprintf(fid, 'format ascii 1.0\n');
    fprintf(fid, 'element face 0\n');
    fprintf(fid, 'property list uchar int vertex_indices\n');
    fprintf(fid, 'element vertex %d\n', npt);
    fprintf(fid, 'property float x\n');
    fprintf(fid, 'property float y\n');
    fprintf(fid, 'property float z\n');
    fprintf(fid, 'property uchar diffuse_red\n');
    fprintf(fid, 'property uchar diffuse_green\n');
    fprintf(fid, 'property uchar diffuse_blue\n');
    fprintf(fid, 'end_header\n');
    
    for i = 1:npt
        fprintf(fid, '%f %f %f\n0 0 0\n', X(i, 1), X(i, 2), X(i, 3));
    end;
    
    fclose(fid);
end;



