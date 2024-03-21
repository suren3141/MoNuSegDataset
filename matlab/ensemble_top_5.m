% Script to combine instance masks of top 5 techniques to get ensemble mask (through instance level majority voting).
% Created by Ruchika Verma, please cite the following paper if you use this code-
% N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane and A. Sethi, 
% "A Dataset and a Technique for Generalized Nuclear Segmentation for 
% Computational Pathology," in IEEE Transactions on Medical Imaging, 
% vol. 36, no. 7, pp. 1550-1560, July 2017

clc;
clear all;
close all;

% Set path
dirname = 'D:\Research work\MONUSEG\Top_5_teams';
addpath('D:\Research work\MONUSEG\Top_5_teams');
listing = dir(dirname);
patient_names = listing(3:end);


% Loop start with top ranker
correct_dirname = 'D:\Research work\MONUSEG\Top_5_teams\1_yanning zhou\Zhou_MoNuSeg2018results';
correct_listing = dir([correct_dirname,'\*.mat']);


% Comparison of each image
destination_path = 'D:\Research work\MONUSEG\Top_5_teams\combined_results' 
mkdir(destination_path);% To save combined maps of top 5 techniques

for j = 1:14 %images

correct_mask = load(strcat(correct_dirname,'\',correct_listing(j).name));
correct_mask = double(cell2mat(struct2cell(correct_mask)));

correct_list = unique(correct_mask); % set of unique nuclei predicted by rank 1 team
correct_list = correct_list(2:end); % exclude 0
ncorrect = numel(correct_list);
predicted_map{1} = correct_mask;
pr_list{1} = correct_list;

% Other Participants

for k = 2:5%Participant
participant = (strcat(dirname,'\',patient_names(k).name));
cd(participant);
participant_name = dir();
participant_name(~[participant_name.isdir]) = [];
participant_name = strcat(participant,'\',participant_name(3).name);

predicted = load(strcat(participant_name,'\',correct_listing(j).name(1:end-4),'_predicted_map.mat'));
predicted_map{k} = double(cell2mat(struct2cell(predicted)));

prList = unique(predicted_map{k}); % ordered set of unique nuclei predicted by rank k team
prList = prList(2:end); % exclude 0
% mark used nuclei by the number of uses (you can use any other criteria) 
pr_list{k} =  prList;
npredicted{k} = numel(prList);
end

% Loop starts
pseudo_mask = zeros(1000,1000); % To save final mask
kk = 1; % Nuclei count

for k = 1:3 %Loop will  cover all nuclei from top 3 techniques
    
correct_list = pr_list{k};
correct_mask = predicted_map{k};

while ~isempty(correct_list)
    fprintf('Processing object # %d \n',kk);
    temp_mask = zeros(1000,1000);
    temp_mask1 = zeros(1000,1000);
    
    temp_mask = (correct_mask==correct_list(1));%ones at the correct instance only
    temp_mask1 = (correct_mask==correct_list(1));%Used to compute pseudo_mask
    correct_mask(correct_mask == correct_list(1)) = 0;
    correct_list(1)= [];
    
    for ll = k+1:5
        pred = temp_mask.*predicted_map{ll};%Has intersecting unique labels 
%         predicted_indices{ll} = nonzeros(unique(pred));
        predicted_indices = nonzeros(unique(pred));
        
        if ~nnz(predicted_indices) == 0%If non-zero, find intersecting pixels
            intersection_pixels = [];
            for i=1:numel(predicted_indices)
               temp = temp_mask.* (pred==predicted_indices(i));
               intersection_pixels(i) = sum(temp(:));      
            end
        
        [n idx]= max(intersection_pixels);
        matched_idx = predicted_indices(idx);
        
        temp_mask1 = temp_mask1+(predicted_map{ll}==matched_idx);
        
        % omit maximum overlapped instance index from the index list and
        % predicted maps
        temp = pr_list{ll};
        temp(temp == matched_idx) = [];
        pr_list{ll} = temp;
        
        temp = predicted_map{ll};
        temp(temp == matched_idx) = 0;
        predicted_map{ll} = temp;
        
        
        end
    end
    
    if nnz(temp_mask1>=3) %threshold is 3
        pseudo_mask = pseudo_mask+kk*(temp_mask1>=3); 
        kk = kk+1;
    end
end

pr_list{k} = correct_list;
predicted_map{k} = correct_mask;

end

% figure;
% imshow(pseudo_mask);

cd (destination_path);
save([correct_listing(j).name(1:end-4),'_combined.mat'],'pseudo_mask');

end
