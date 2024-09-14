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

% check for missing values
missing = ismissing(data);

disp(data( any(missing, 2), :));

%%
data.Properties.VariableNames = strrep(data.Properties.VariableNames, '_', '');

% histograms of Variables
figure;  
for i = 2:size(data,2)
    subplot(5, 5, i-1);  
    histogram(data{:,i});
    title(['Histogram of ', data.Properties.VariableNames{i}]);
    ylabel('Frequency');
end

% boxplot
figure; 
for i = 2:size(data,2)
    subplot(5, 5, i-1);  
    boxplot(data{:,i});
    title(['Boxplot of ', data.Properties.VariableNames{i}]);
    ylabel(data.Properties.VariableNames{i})
end

%%
