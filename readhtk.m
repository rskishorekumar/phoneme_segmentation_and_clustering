function [m,nSamples,sampPeriod,sampSize,paramKind]=readhtk (name,arch); 
% [m,nSamples,sampPeriod,sampSize,paramKind]=readhtk (filename,arch); 
%
% Simple function to read an HTK file. If arch is 'n', the native machine
% format will be used, otherwise HTK deafult (BIG endian) is expected. 
% accepts only FLOAT vectors 
if (nargin ==1)
  arch = 'b';
end

ff = fopen (name,'r',arch);
% read header
nSamples = fread (ff,[1 1],'int');
sampPeriod = fread (ff,[1 1],'int');
sampSize = fread (ff,[1 1],'short');
paramKind = fread (ff,[1 1],'short');
% determine amount of data to read and read 
P = sampSize / 4; 
m = fread (ff, [P nSamples], 'float');
fclose (ff);