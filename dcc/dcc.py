% Script to use Martin Lindquist's dynamic connectivity toolbox to create
% correlation matrices at each point in time instead of across an entire
% block or chunk of a block
% See /home/despo/jrcohen/matlab/DC_toolbox/Example.m
% Made by JRC 9-1-14

%%%%% NOTE: MUST USE MATLAB 2014a!!!!! %%%%%

clear all;

% Define paths/variables
basedir = '/home/despo/kki_adhd/';
scriptdir = [basedir '/scripts/'];
analdir = [basedir '/analyses/03_rest_corr_4-22-13_KKIproc_restonly/'];
corrdir = 'corr_';
atlases={'aal'}; % 'aal',dosenbach_all'
mindist=20; % If using a mask to 0 out short connections
corrtypedir = '/roi2roi/';
corrmatdir = '/DC_TXTfiles/';
timesdir = '/timeseries/';
windowsize = 20; % If do sliding window correlations

% Check for version and exit if not using 2014a
 v=version('-release');
 if ~strcmp(v,'2014a'),
    fprintf('\n*************************\nMUST USE MATLAB 2014a!!!!\n*************************\n\n');
    return
 end;

task=input('Is this analysis for all rest subs (1), task subs, rest blocks (2), or task subs, task blocks (3)? ');
while isempty(find(task==[1 2 3])),
    task=input('Must enter 1,2, or 3--please re-enter: ');
end;

if task==1, % Rest
    analdir = [basedir '/analyses/03_rest_corr_4-22-13_KKIproc_restonly/'];
elseif task==2, % Rest for task
    analdir = [basedir '/analyses/06_rest_corr_10-6-14_KKIproc_restfortask/'];
elseif task==3, % Task only
    analdir = [basedir '/analyses/06_rest_corr_10-6-14_KKIproc_taskonly/'];
end;

tstart=tic; % So can timestamp how long everything takes, with toc(tstart)

for a=1:length(atlases),
    atlas=atlases{a};
    if strcmp(atlas,'dosenbach_all'), % Load mask to 0 out short connections
        anatomdist_mask=load([basedir '/masks/norm_' atlas '/masks_anatomdist/' atlas '_mindist' num2str(mindist) '.txt']);
    else,
        clear anatomdist_mask % Clear if not dos_all to not risk accidentally masking for the wrong atlas
    end;
    fulldcdir=[analdir corrdir atlas corrtypedir corrmatdir]; % Output dir for dynamic connectivity correlation matrices
    if ~exist(fulldcdir),
        system(['mkdir ' fulldcdir]);
    end;
    fulltimesdir=[analdir corrdir atlas corrtypedir timesdir];
    timesfiles=dir([fulltimesdir,'*mat']);
    for t=1:length(timesfiles),
    %for t=190:length(timesfiles), % Running all subs after one that failed
    %for t=[125 195 202], % Short for trouble-shooting
        tstart2=tic; % So can timestamp how long everything takes, with toc(tstart)    
        tname=timesfiles(t).name;
        tmp=strsplit(tname,'_');
        sub=tmp{1};
        sess=tmp{2};
        block=tmp{3};
        fprintf('\n=====\n%s\n=====\n',sub);
        pause(1); % Insert a pause so I can see the time and what sub we're on
        tseries=load([fulltimesdir,tname]);
        data=tseries.data;
        % Define data dimensions
        p = size(data,2); % Number of nodes
        T = size(data,1); % Number of time points
        data_stand=zscore(data); % Standardize data before run DCC
        % Check for ROIs with all 0s (meaning they don't exist), and 
        % remove them temporarily for DCC calc
        empty=[];
        for roi=1:size(data_stand,2),
            if sum(data_stand(:,roi))==0, 
                %fprintf('%d\n',roi); % To make sure indexing correctly
                empty=cat(2,empty,roi); % Keep a list of empty ROIs
            end;
        end;
        data_stand(:,empty)=[];
        % Fit DCC
        [corrmat] = mvDCC(data_stand); % fits DCC separately for each pair of nodes...models variance more accurately than calling DCC
        %[corrmat, covmat] = DCC(data_stand); % corrmat = dynamic correlation matrix; covmat = dynamic covariance matrix
        % Fit sliding window correlations--NOT DOING THIS FOR NOW!!
        %[sw_corrmat] = sliding_window(data_stand,windowsize); % sw_corrmat = sliding window correlation matrix 
        % Save output correlation matrix--each timepoint separately
        for n=1:size(corrmat,3), 
            % Save each timepoint separately
            if task==3, % Task only
                if strcmp(block,'Block01'), % Run 1, start count at 0
                    corrmatname=sprintf('%s%s_%s_Block%03d.txt',fulldcdir,sub,sess,n);
                elseif strcmp(block,'Block02'), % Run 2, start count at size(corrmat,3)+1
                    corrmatname=sprintf('%s%s_%s_Block%03d.txt',fulldcdir,sub,sess,size(corrmat,3)+n);
                end;
            else, % Rest only and rest for task both only 1 sess/run
                corrmatname=sprintf('%s%s_%s_Block%03d.txt',fulldcdir,sub,sess,n);
            end;
            %fprintf('%s\n',corrmatname); % To make sure I'm naming correctly
            corrmat_1vol=corrmat(:,:,n);
            % Re-insert any ROIs that were temporarily removed for DCC calc (with values of NaN)
            % First fill in empty rows
            corrmat_fullrows_1vol=NaN(p,size(corrmat_1vol,2));
            count=1;
            for r=1:p,
                if isempty(find(empty==r)),
                    corrmat_fullrows_1vol(r,:)=corrmat_1vol(count,:);
                    count=count+1;
                end;
            end;
            % Second fill in empty columns
            corrmat_full_1vol=NaN(p,p);
            count=1;
            for c=1:p,
                if isempty(find(empty==c)),
                    corrmat_full_1vol(:,c)=corrmat_fullrows_1vol(:,count);
                    count=count+1;
                end;
            end;
            if exist('anatomdist_mask','var'), % If loaded distance mask, use it to 0 out short connections
                corrmat_masked=corrmat_full_1vol.*anatomdist_mask;
            else,
                corrmat_masked=corrmat_full_1vol;
            end;
            save(corrmatname,'corrmat_masked','-ascii');
        end;
        fprintf('    Time elapsed for %s: %.2f minutes for sub, %.2f minutes total.\n',sub,toc(tstart2)/60,toc(tstart)/60);
        pause(1); % Insert a pause so I can see the time and what sub we're on
    end; % End timesfile loop
end; % End atlas loop