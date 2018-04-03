base_dir = 'C:\Users\fango\Google Drive\Packt\Camera_Calibration\Camera_Calibration\Patterns';
base_dir = strrep (base_dir, '\', '/');

allImg   = dir([base_dir '/*.jpg']);
allNames = {allImg.name};
%% ===== Write image_list.txt ======= %%
listID = fopen([base_dir '/image_list.txt'], 'w');
for c = 1 : size(allNames,2)
    fprintf(listID, '%s\n', [base_dir '/' allNames{c}]);
end
fclose(listID);