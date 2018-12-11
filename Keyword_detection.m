%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Unsupervised pattern discovery in speech utterances
% using phoneme segmenation and depth first search approach.

% Author: R Kishore Kumar
% Email : rskishorekumar@gmail.com
% Institute : Indian Institute of Technology Kharagpur
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all;
clc;


%/////////////////////////////////////////////////////////////////////////
%% Calculate (or load) signal features at 100 Hz sampling rate

fprintf('extracting features from HTK...');
fprintf(' Done!\n');
n_signals = 2; % 68 corresponds to the full set, 5 to the first talker etc.
n_cuts = 5;            % How many sub-segments per syllable? (default: 5)
folder = 'WAV/';       % Location of audio files

%/////////////////////////////////////////////////////////////////////////
%% Get signal filenames for each signal in the speech corpus

% Get filenames for audio files
a = dir([folder '*.wav']);
Signal_filename = cell(length(a),1);
for k = 1:length(a)
    Signal_filename{k} = [folder a(k).name];
    procbar(k,n_signals);
end

fprintf('\n');
fprintf(' Changing the signal filenames into normal order!\n');
[signal_filenames,INDEX] = sort_nat(Signal_filename);
fprintf('\n');
fprintf(' Done!\n');
fprintf('\n');

%%/////////////////////////////////////////////////////////////////////////
%% Extract the Posterior feature vector for all the wav files in /WAV/ Folder

% load('dpmm_142_100it.mat'); load this file after training the MFCC files
% with DPGMM 

folder = 'MFCC/'; 
a = dir([folder '*.mfcc']);

signal_filenames_posteriors = cell(length(a),1);
signal_filenames_clusters = cell(length(a),1);
fprintf('\n');
fprintf(' Changing the signal filenames into normal order!\n');
fprintf('\n');
fprintf(' Done!\n');
fprintf('\n');

lamda=0.0001;
ud = 0.1270;
m=2;
j=0;
k=1;
for i=1:m
    filename = signal_filenames{i,1};
    [filepath,name,ext] = fileparts(filename);
        
    fnpost=strcat('MFCC/',strrep(name, '.mfcc', '.post'),'.mfcc');
    feat=readhtk(char(fnpost));
    l=length(feat);
    l=j+l;
    feat = dpmm_posterior((j+1):l,:);
    tfeat=(1-lamda)*feat+lamda*ud;
    signal_filenames_posteriors{k} = tfeat;
    j=l;
    k=k+1;
    procbar(i,m);
end
fprintf('\n');
fprintf(' Done!\n');

%/////////////////////////////////////////////////////////////////////////
%% Extract the amplitude for all the wav files in /WAV/ Folder


% feat_comps_to_take = 1:39; % Which coefficients of the original features to include

SA = cell(length(signal_filenames),1);
N = length(signal_filenames);
feat1=[];
%/////////////////////////////////////////////////////////////////////////
%Placing the MFCC along the corresponding signal files.

for k = 1:length(signal_filenames)   
    
    filename = signal_filenames{k};
       
    [qa1,qfs] = audioread(char(filename));
    
    SA{k} = qa1;
    procbar(k,length(signal_filenames));
    
end
fprintf('\n');

fprintf(' Done!\n');

%/////////////////////////////////////////////////////////////////////////
%% Extract the MFCC feature vector for all the wav files in /WAV/ Folder


% feat_comps_to_take = 1:39; % Which coefficients of the original features to include

M = cell(length(signal_filenames),1);
N = length(signal_filenames);
feat1=[];
%/////////////////////////////////////////////////////////////////////////
%Placing the MFCC along the corresponding signal files.

for k = 1:length(signal_filenames)   
    
    filename = signal_filenames{k};
    [filepath,name,ext] = fileparts(filename);
    
    fnpost=strcat('MFCC/',strcat(name,'.mfcc'));
    feat=readhtk(char(fnpost));
    
    M{k} = feat';
    procbar(k,N);
    
end
fprintf('\n');

fprintf(' Done!\n');

%/////////////////////////////////////////////////////////////////////////
%% Segmenting the scilence portions in the speech wav file based on energy.  
%  Convert the posteriors into the corresponding file size 

fprintf('Segmenting the wav file based on the energy.\n');
SSSS = cell(n_signals,1); 
Senergy = cell(n_signals,1); 
Ssequence = cell(n_signals,1); 

Fs = 16000;
Frame_size = 25;  %Input: Frame-size in millisecond
Frame_shift = 10; %Input: Frame-shift in millisecond
window_period=Frame_size/1000;
shift_period=Frame_shift/1000;

window_length = window_period*Fs;
sample_shift = shift_period*Fs;

for signal = 1:n_signals
    for i=1:(floor((length(SA{signal,1}))/sample_shift)-ceil(window_length/sample_shift))
        k=1;yy=0;
        for j=(((i-1)*sample_shift)+1):(((i-1)*sample_shift)+window_length)
            yy(k)=SA{signal,1}(j);
            k=k+1;
        end
        Senergy{signal,1}{i,1} = round(sum(abs(yy.*yy)),1);
        Ssequence{signal,1}{i,1} = i;
    end
     procbar(signal,n_signals);
end
fprintf('\n');

fprintf(' Done!\n');

threshold = 25;
energythreshold = 0.1;
ModiM = cell(n_signals,1); 
ModiP = cell(n_signals,1); 
ModiS = cell(n_signals,1); 
ModiE = cell(n_signals,1); 


for signal = 1:n_signals
    k=1;
    kk=1;
    while k <= length(Senergy{signal})

             
        if(Senergy{signal,1}{k,1} <= energythreshold)
            % Get index for starting point
            local_start = k;
            cc = local_start;
          
            while (Senergy{signal,1}{cc,1} <= energythreshold)
                local_end = cc;
                cc = cc + 1;
                if(cc >= length(Senergy{signal}))
                    break;
                end
            end
            
            if (local_end - local_start >= threshold)
                % Get the vectors of only the voiced portions
                k = local_end;
            end
        end
         ModiM{signal}(kk,:) = M{signal}(k,:);
         ModiP{signal}(kk,:) = signal_filenames_posteriors{signal}(k,:);
         ModiS{signal}{kk,1} = Ssequence{signal}{k,1};
         ModiE{signal}{kk,1} = Senergy{signal}{k,1};
         k = k + 1;
         kk = kk + 1;
    end
    procbar(signal,n_signals);
end
fprintf('\n');

fprintf(' Done!\n');
 

ModiMM = cell(n_signals,1); 
ModiPP = cell(n_signals,1); 
ModiSS = cell(n_signals,1); 
ModiEE = cell(n_signals,1); 


for signal = 1:n_signals
    matcell=cell2mat(ModiE{signal,1});
    B = (matcell == 0);
    i=1;
    frontscilence=1;
    while(B(i)==1)
        frontscilence=i;
        i=i+1;
    end
    last=length(B);
    i=last;
    lastscilence=length(B);
    while(B(i)==1)
        lastscilence=i;
        i=i-1;
    end
    k=frontscilence;
    kk=1;
    while(k <= lastscilence)
        ModiMM{signal}(kk,:) = ModiM{signal}(k,:);
        ModiPP{signal}(kk,:) = ModiP{signal}(k,:);
        ModiSS{signal}{kk,1} = ModiS{signal}{k,1};
        ModiEE{signal}{kk,1} = ModiE{signal}{k,1};
        k=k+1;
        kk=kk+1;
    end
    procbar(signal,n_signals);
end
fprintf('\n');

fprintf(' Done!\n');


%/////////////////////////////////////////////////////////////////////////
%% Finding the phoneme-like units using MFCC/posterior feature of all the wav

% Compute the affinity matrix and find the phoneme boundaries
% based of the 4th frame and median filter


boundaries = cell(n_signals,1);

for signal = 1:length(boundaries)
          
    feat = ModiMM{signal};
    A = feat;
    B = A';
    AFF = A * B;
    nc=size(AFF,2);         % number of columns
    ix=zeros(1,nc);       % preallocate for the index

    % Extract only the diagonal values
    for i=1:nc-3
        ix(i)=AFF(i,i+3);  
    end

    y = medfilt1(ix,4);
    TF = islocalmin(y);
    boundaries{signal} = TF;
    procbar(signal,length(boundaries));
end
fprintf('\n');
fprintf('Done!\n');


%/////////////////////////////////////////////////////////////////////////
%% Segmenting the feature file based on the phone segments. 
%  Convert the posteriors into the corresponding file size 

fprintf('Segmenting the feature file based on the segments.\n');
span = 0;       % How many frames to include outside segment boundaries
S = cell(n_signals,1); 
SE = cell(n_signals,1); 

feat_comps_to_take = 1:142; % Which coefficients of the original features to include
feat_comps_to_takes = 1;

for signal = 1:n_signals
    boundaries{signal,1}(1,length(boundaries{signal})) = 1;
    kk = find(boundaries{signal, 1});
    S{signal} = cell(length(kk),1);
    n = 1;
    S{signal}{n} = ModiPP{signal}(max(1,1):min(kk(1)-1,size(ModiPP{signal},1)),feat_comps_to_take);
    SE{signal}{n} = ModiSS{signal}(max(1,1):min(kk(1)-1,size(ModiSS{signal},1)),feat_comps_to_takes);
    n=n+1;
    for k = 1:length(kk)-1
        
        if (k==(length(kk)-1))
            S{signal}{n} = ModiPP{signal}(max(1,kk(k)):min(kk(k+1),size(ModiPP{signal},1)),feat_comps_to_take);
            SE{signal}{n} = ModiSS{signal}(max(1,kk(k)):min(kk(k+1),size(ModiSS{signal},1)),feat_comps_to_takes);
        else
            S{signal}{n} = ModiPP{signal}(max(1,kk(k)):min(kk(k+1)-1,size(ModiPP{signal},1)),feat_comps_to_take);
            SE{signal}{n} = ModiSS{signal}(max(1,kk(k)):min(kk(k+1)-1,size(ModiSS{signal},1)),feat_comps_to_takes);
        end
        
        n = n+1;
        
    end
    
     procbar(signal,N);
     clear kk;
end
fprintf('\n');

fprintf(' Done!\n');

%/////////////////////////////////////////////////////////////////////////
%% Convert the posteriors into the equal size in each frame
% fixed length feature vectors by dividing the phoneme into N sub-segments
% and averaging feature vectors over those segments, finally concatenating
% all N representations into one long feature vector.

n_cuts = 5;
f = cell(n_signals,1); % <-- this is where phoneme feature vectors go

featlen = size(S{1}{1},2);

fprintf(' Concatenating features !\n');

for signal = 1:n_signals
    f{signal} = zeros(length(S{signal}),n_cuts*featlen+1);
    
    for k = 1:length(S{signal})
        
        % Divide into N sub-segments uniformly in time
        
        len = size(S{signal}{k},1);
        
        if(len > 1)
            % Get sub-segment boundaries
            local_bounds = zeros(n_cuts+1,1);
            local_bounds(1) = 1;
            local_bounds(end) = len;
            for cc = 1:n_cuts-1
                local_bounds(cc+1) = max(1,round(cc/n_cuts*len));
            end
            
            % Get mean features across sub-segments
            for t = 1:length(local_bounds)-1
                f{signal}(k,(t-1)*featlen+1:t*featlen) = mean(S{signal}{k}(local_bounds(t):local_bounds(t+1),:));
            end
            f{signal}(k,end) = log(len).*n_cuts/3;      % Add scaled log-duration
         end
    end
    procbar(signal,n_signals);
end
fprintf('\n');

fprintf(' Done!\n');


%/////////////////////////////////////////////////////////////////////////
%% code with distance matrix computation with two phoneme relaxation and done the preprocessing of scilence removal

load('data_phoneme_features.mat');
termFN='2Proposed_5_142';
termfID = fopen(termFN,'w');

D=[];
flag = 0;
n_signals = 2;
min_interval_length = 5;
cost_threshold = 0.5;
cost_array = zeros(1,1);

for q=1:n_signals
    for t=(q+1):n_signals
        D=real(newnomKLSDiv(f{q,1}',f{t,1}'));     
   
        qlen=size(D,1);
        tlen=size(D,2);
        mark = zeros(qlen,tlen);
        outer_flag = 0;
        for i=1:qlen
            for j=1:tlen
                if mark(i,j)==1
                    continue;
                end
                if D(i,j) <= cost_threshold
                    cost=D(i,j);
                    k=1;
                    excess_flag = 0;
                    flag = 0;
                    diagonal_flag = 0;
                    i1=0;
                    j1=0;
                    cost_array = zeros(min_interval_length,1);
                    cost_array(1) = cost;
                    printflag = 0;
                    while(cost<=cost_threshold && i+k+i1 <= qlen && j+k+j1 <= tlen)
                        
                        
                        % checking for diagonal or other directions part
                        if diagonal_flag == 0
                            
                            cost1=D(i+k+i1,j+k-1+j1);
                            cost2=D(i+k-1+i1,j+k+j1);
                            cost3=D(i+k+i1,j+k+j1);
                            
                            if cost3<=cost_threshold
                                cost = cost3;
                                mark(i+k+i1,j+k+j1)=1;
                            else
                                if cost3<=cost2 && cost3<=cost1
                                    %diagonal
                                    diagonal_flag=0;
                                    cost = cost3;
                                    mark(i+k+i1,j+k+j1)=1;
                                    
                                elseif cost2<=cost3 && cost2<=cost1
                                    %vertical
                                    cost = cost2;
                                    diagonal_flag = 1;
                                    mark(i+k-1+i1,j+k+j1)=1;
                                    j1 = j1+1;
                                    k = k-1;
                                else
                                    %horizontal
                                    cost = cost1;
                                    diagonal_flag = 1;
                                    mark(i+k+i1,j+k-1+j1)=1;
                                    i1 = i1+1;
                                    k = k-1;
                                end
                            end
                            
                            
                        else
                            cost=D(i+k+i1,j+k+j1);
                            diagonal_flag=0;
                            mark(i+k+i1,j+k+j1)=1;
                        end
                        
                        cost_array(k+1) = cost;
  
                        
                        if (k>=min_interval_length && printflag == 0)
                            cost_avg=0;
                            count_cost=1;
                            for l=1:min_interval_length
                               cost_avg=cost_avg+cost_array(l);
                               if (cost_array(l)>=0.12)
                                   count_cost=count_cost+1;
                               end
                            end
                            if(count_cost<=2)
                                %fprintf("doc %d phone %d : doc %d phone %d i1=%d j1=%d k=%d ",q,i,t,j,i1,j1,k);
                                fprintf(termfID,"doc %d phone %d : doc %d phone %d i1=%d j1=%d k=%d ",q,i,t,j,i1,j1,k);
                                for l=1:min_interval_length
                                    %fprintf("C(%d)= %f ",l,cost_array(l));
                                    fprintf(termfID,"C(%d)= %f ",l,cost_array(l));
                                end
                                %fprintf("C_avg= %f \n",cost_avg);
                                fprintf(termfID,"C_avg= %f \n",cost_avg);
                            end
                            printflag = 1;
                        end
                        k=k+1;
                        cost_array(k) = cost;
                    end
                end
            end
        end
    end
    procbar(q,n_signals);
end

fclose(termfID);

% In the 2Proposed_5_142 file the matched keywords regions are shown.
%\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
