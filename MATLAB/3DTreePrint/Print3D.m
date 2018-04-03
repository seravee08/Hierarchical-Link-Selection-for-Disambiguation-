%% ================ Print 3D Camera Linkage =================== %%
nvm_base = 'L:/Data/Images/ImageSet/top_top/D16.66';
nvm_dir  = [nvm_base, '/model.nvm'];
mach_dir = [nvm_base, '/matchings_ultra.txt'];
R9T_Format = false;
%% ================ Change Parameters ==================== %%
nvm_file = fopen(nvm_dir, 'r');
dum      = textscan(nvm_file, '%s', 1, 'Delimiter', '\n');
cam_num  = textscan(nvm_file, '%d', 1, 'Delimiter', '\n');
if cam_num{1}(1) <= 0
    cam_num = textscan(nvm_file, '%d', 1, 'Delimiter', '\n');
end
%% ================ Readin R and T in different formats ==================%%
P = zeros(3 * cam_num{1}(1), 4);
if R9T_Format == true
    cam_info = textscan(nvm_file, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f', cam_num{1}(1));
    for i = 1 : 1 : cam_num{1}(1)
        P((i-1)*3+1,1) = cam_info{3}(i); P((i-1)*3+1,2) = cam_info{4}(i);
        P((i-1)*3+1,3) = cam_info{5}(i); P((i-1)*3+1,4) = cam_info{12}(i);
        P((i-1)*3+2,1) = cam_info{6}(i); P((i-1)*3+2,2) = cam_info{7}(i);
        P((i-1)*3+2,3) = cam_info{8}(i); P((i-1)*3+2,4) = cam_info{13}(i);
        P((i-1)*3+3,1) = cam_info{9}(i); P((i-1)*3+3,2) = cam_info{10}(i);
        P((i-1)*3+3,3) = cam_info{11}(i); P((i-1)*3+3,4) = cam_info{14}(i);
    end
else
    cam_info = textscan(nvm_file, '%s %f %f %f %f %f %f %f %f %f %f', cam_num{1}(1));
    for i = 1 : 1 : cam_num{1}(1)
        quat = [cam_info{3}(i) cam_info{4}(i) cam_info{5}(i) cam_info{6}(i)];
        tran = [cam_info{7}(i) cam_info{8}(i) cam_info{9}(i)];
        rotm = quat2rotm(quat);
        rotm = rotm';
        tran = -rotm' * tran';
        rotm = rotm';
        P((i-1)*3+1,1) = rotm(1,1); P((i-1)*3+1,2) = rotm(1,2);
        P((i-1)*3+1,3) = rotm(1,3); P((i-1)*3+1,4) = tran(1);
        P((i-1)*3+2,1) = rotm(2,1); P((i-1)*3+2,2) = rotm(2,2);
        P((i-1)*3+2,3) = rotm(2,3); P((i-1)*3+2,4) = tran(2);
        P((i-1)*3+3,1) = rotm(3,1); P((i-1)*3+3,2) = rotm(3,2);
        P((i-1)*3+3,3) = rotm(3,3); P((i-1)*3+3,4) = tran(3);
    end
end
dum     = textscan(nvm_file, '%d', 1, 'Delimiter', '\n');
pt_num  = textscan(nvm_file, '%d', 1, 'Delimiter', '\n');
if pt_num{1}(1) <= 0
    pt_num = textscan(nvm_file, '%d', 1, 'Delimiter', '\n');
end
pt    = zeros(pt_num{1}(1), 3);
clr = zeros(pt_num{1}(1), 3); 
for i = 1 : 1 : pt_num{1}(1)
    pt_info = textscan(nvm_file, '%f %f %f %d %d %d %d', 1, 'Delimiter', '\n');
    measurement_num = pt_info{7}(1);
    dum = textscan(nvm_file, '%f', measurement_num * 4);
    dum = textscan(nvm_file, '\n');
    pt(i, 1) = pt_info{1}(1);
    pt(i, 2) = pt_info{2}(1);
    pt(i, 3) = pt_info{3}(1);
    clr(i, 1) = pt_info{4}(1);
    clr(i, 2) = pt_info{5}(1);
    clr(i, 3) = pt_info{6}(1);
end
fclose(nvm_file);
%% ================ Read in matchings.txt =================== %%
mach_file = fopen(mach_dir, 'r');
pair = [];
while ~feof(mach_file)
    nameL = textscan(mach_file, '%s', 1, 'Delimiter', '\n');
    nameR = textscan(mach_file, '%s', 1, 'Delimiter', '\n');
    nL = nameL{1}(1); nL = nL{1,1};
    nR = nameR{1}(1); nR = nR{1,1};
    idxL = -1; idxR = -1;
    for i=1:cam_num{1}(1)
        dirL = [nvm_base, '/'];
        tmp  = cam_info{1}(i);
        tmp = tmp{1,1};
        dirL = [dirL, tmp];
        if dirL == nL
            idxL = i;
            break;
        end
    end
    for i=1:cam_num{1}(1)
        dirR = [nvm_base, '/'];
        tmp = cam_info{1}(i);
        tmp = tmp{1,1};
        dirR = [dirR, tmp];
        if dirR == nR
            idxR = i;
            break;
        end
    end
    if idxL ~= -1 && idxR ~= -1
        pair = cat(1, pair, [idxL idxR]);
    end
    dum = fgets(mach_file);
    dum = fgets(mach_file);
    dum = fgets(mach_file);
    dum = fgets(mach_file);
end
fclose(mach_file);
%% ================ Call Nianjuan function ================== %%
CameraMatrixPoints2OBJ(P, pt, clr, pair); %% New Version
%% CameraMatrixPoints2OBJ_Orin(P, pt); %% Oringal Version