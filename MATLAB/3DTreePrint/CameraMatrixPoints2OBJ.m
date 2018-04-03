function [] = CameraMatrixPoints2OBJ(P, X, clr, pair)

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
       if min_dist == 0 || min_dist > dist
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
    pt_stash = zeros(ncam*13, 3);
    for i = 1:ncam
        %% ===== Scale the camera ===== %%
        tip    = campts(1, :, i);
        curCam = campts(:, :, i) * 7;
        tipDif = curCam(1, :) - tip;
        for j = 1:5
            curCam(j,:) = curCam(j,:) - tipDif;
        end
        fprintf(fid, 'v %f %f %f c %d %d %d\n', curCam(1, 1), curCam(1, 2), curCam(1, 3), -255, 120, 120);
        fprintf(fid, 'v %f %f %f c %d %d %d\n', curCam(2, 1), curCam(2, 2), curCam(2, 3), -255, 120, 120);
        fprintf(fid, 'v %f %f %f c %d %d %d\n', curCam(3, 1), curCam(3, 2), curCam(3, 3), -255, 120, 120);
        fprintf(fid, 'v %f %f %f c %d %d %d\n', curCam(4, 1), curCam(4, 2), curCam(4, 3), -255, 120, 120);
        fprintf(fid, 'v %f %f %f c %d %d %d\n', curCam(5, 1), curCam(5, 2), curCam(5, 3), -255, 120, 120);
        pt_stash((i-1)*13+1,1) = curCam(1, 1); pt_stash((i-1)*13+1,2) = curCam(1, 2); pt_stash((i-1)*13+1,3) = curCam(1, 3);
        pt_stash((i-1)*13+2,1) = curCam(2, 1); pt_stash((i-1)*13+2,2) = curCam(2, 2); pt_stash((i-1)*13+2,3) = curCam(2, 3);
        pt_stash((i-1)*13+3,1) = curCam(3, 1); pt_stash((i-1)*13+3,2) = curCam(3, 2); pt_stash((i-1)*13+3,3) = curCam(3, 3);
        pt_stash((i-1)*13+4,1) = curCam(4, 1); pt_stash((i-1)*13+4,2) = curCam(4, 2); pt_stash((i-1)*13+4,3) = curCam(4, 3);
        pt_stash((i-1)*13+5,1) = curCam(5, 1); pt_stash((i-1)*13+5,2) = curCam(5, 2); pt_stash((i-1)*13+5,3) = curCam(5, 3);
        %% ============= Contrsruct a cube =============== %%
        fprintf(fid, 'v %f %f %f c %d %d %d\n', curCam(1, 1)+f/5, curCam(1, 2)+f/5, curCam(1, 3)+f/5, 255, 0, -200); %a1
        fprintf(fid, 'v %f %f %f c %d %d %d\n', curCam(1, 1)+f/5, curCam(1, 2)-f/5, curCam(1, 3)+f/5, 255, 0, -200); %a2
        fprintf(fid, 'v %f %f %f c %d %d %d\n', curCam(1, 1)-f/5, curCam(1, 2)-f/5, curCam(1, 3)+f/5, 255, 0, -200); %a3
        fprintf(fid, 'v %f %f %f c %d %d %d\n', curCam(1, 1)-f/5, curCam(1, 2)+f/5, curCam(1, 3)+f/5, 255, 0, -200); %a4
        fprintf(fid, 'v %f %f %f c %d %d %d\n', curCam(1, 1)+f/5, curCam(1, 2)+f/5, curCam(1, 3)-f/5, 255, 0, -200); %a5
        fprintf(fid, 'v %f %f %f c %d %d %d\n', curCam(1, 1)+f/5, curCam(1, 2)-f/5, curCam(1, 3)-f/5, 255, 0, -200); %a6
        fprintf(fid, 'v %f %f %f c %d %d %d\n', curCam(1, 1)-f/5, curCam(1, 2)-f/5, curCam(1, 3)-f/5, 255, 0, -200); %a7
        fprintf(fid, 'v %f %f %f c %d %d %d\n', curCam(1, 1)-f/5, curCam(1, 2)+f/5, curCam(1, 3)-f/5, 255, 0, -200); %a8
        pt_stash((i-1)*13+6,1)  = curCam(1, 1)+f/5; pt_stash((i-1)*13+6,2)  = curCam(1, 2)+f/5; pt_stash((i-1)*13+6,3)  = curCam(1, 3)+f/5;
        pt_stash((i-1)*13+7,1)  = curCam(1, 1)+f/5; pt_stash((i-1)*13+7,2)  = curCam(1, 2)-f/5; pt_stash((i-1)*13+7,3)  = curCam(1, 3)+f/5;
        pt_stash((i-1)*13+8,1)  = curCam(1, 1)-f/5; pt_stash((i-1)*13+8,2)  = curCam(1, 2)-f/5; pt_stash((i-1)*13+8,3)  = curCam(1, 3)+f/5;
        pt_stash((i-1)*13+9,1)  = curCam(1, 1)-f/5; pt_stash((i-1)*13+9,2)  = curCam(1, 2)+f/5; pt_stash((i-1)*13+9,3)  = curCam(1, 3)+f/5;
        pt_stash((i-1)*13+10,1) = curCam(1, 1)+f/5; pt_stash((i-1)*13+10,2) = curCam(1, 2)+f/5; pt_stash((i-1)*13+10,3) = curCam(1, 3)-f/5;
        pt_stash((i-1)*13+11,1) = curCam(1, 1)+f/5; pt_stash((i-1)*13+11,2) = curCam(1, 2)-f/5; pt_stash((i-1)*13+11,3) = curCam(1, 3)-f/5;
        pt_stash((i-1)*13+12,1) = curCam(1, 1)-f/5; pt_stash((i-1)*13+12,2) = curCam(1, 2)-f/5; pt_stash((i-1)*13+12,3) = curCam(1, 3)-f/5;
        pt_stash((i-1)*13+13,1) = curCam(1, 1)-f/5; pt_stash((i-1)*13+13,2) = curCam(1, 2)+f/5; pt_stash((i-1)*13+13,3) = curCam(1, 3)-f/5;
    end;
    
    for i = 1:ncam
        for j = 1:5
            p(j) = j + (i-1)*13;
        end;
        
        for j = 1:4
            fprintf(fid, 'f %d %d %d\n', p(1), p(rem(j-1, 4) + 2), p(rem(j, 4)+2));
        end;
        fprintf(fid, 'f %d %d %d\n', p(2), p(4), p(3));
        fprintf(fid, 'f %d %d %d\n', p(2), p(5), p(4));
    end;
    
    for o_i=1:size(pair,1)
        idx1 = pair(o_i,1);
        idx2 = pair(o_i,2);
        %% ==== 1st Cube ==== %%
        cubeA = zeros(1,8);
        cubeA(1) = (idx1-1)*13+6;
        cubeA(2) = (idx1-1)*13+7;
        cubeA(3) = (idx1-1)*13+8;
        cubeA(4) = (idx1-1)*13+9;
        cubeA(5) = (idx1-1)*13+10;
        cubeA(6) = (idx1-1)*13+11;
        cubeA(7) = (idx1-1)*13+12;
        cubeA(8) = (idx1-1)*13+13;
        %% ==== 2nd Cube ==== %%
        cubeB = zeros(1,8);
        cubeB(1) = (idx2-1)*13+6;
        cubeB(2) = (idx2-1)*13+7;
        cubeB(3) = (idx2-1)*13+8;
        cubeB(4) = (idx2-1)*13+9;
        cubeB(5) = (idx2-1)*13+10;
        cubeB(6) = (idx2-1)*13+11;
        cubeB(7) = (idx2-1)*13+12;
        cubeB(8) = (idx2-1)*13+13;
        %% ==== Draw Polygon ==== %%
        for i=1:7
            for j=i+1:8
                sameCounter = 0;
                for k=1:3
                    if pt_stash(cubeA(i),k) == pt_stash(cubeA(j),k)
                        sameCounter = sameCounter + 1;
                    end
                end
                if sameCounter == 2
                    for i_i=1:7
                        for i_j=i_i+1:8
                            i_sameCounter = 0;
                            for i_k=1:3
                                if pt_stash(cubeB(i_i), i_k) == pt_stash(cubeB(i_j), i_k)
                                    i_sameCounter = i_sameCounter + 1;
                                end
                            end
                            if i_sameCounter == 2
                                fprintf(fid, 'f %d %d %d\n', cubeA(i), cubeB(i_i), cubeB(i_j));
                                fprintf(fid, 'f %d %d %d\n', cubeA(i), cubeA(j), cubeB(i_j));
                            end
                        end
                    end
                end
            end
        end
    end
    
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
        fprintf(fid, '%f %f %f\n%d %d %d\n', X(i, 1), X(i, 2), X(i, 3),...
            clr(i, 1), clr(i, 2), clr(i, 3));
    end;
    
    fclose(fid);
end;



