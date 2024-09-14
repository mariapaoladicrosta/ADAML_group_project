clear all, close all, clc

% loading the file
data = readtable('/Users/mariapaoladicrosta/Downloads/MiningProcess_Flotation_Plant_Database.csv');

%%
% exploring data
whos data

size(data)

head(data)

summary(data)

%%
% transform cell array of character vectors into double arrays 
for i = 1:size(data, 2)
    if iscell(data{:, i})  % Check if the column is a cell array
        data.(data.Properties.VariableNames{i}) = str2double(strrep(data{:,i}, ',', '.'));
    end
end

summary(data)

%%
