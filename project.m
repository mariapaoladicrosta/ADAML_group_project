clear all;
close all;
clc;

% loading the file
data = readtable('C:\Users\maryamata\OneDrive - KamIT 365\Liitteet\Advance machine learning\archive\MiningProcess_Flotation_Plant_Database.csv');

%% Exploring data
whos data

size(data)

head(data)

% Check the variable names to identify the correct columns
disp(data.Properties.VariableNames'); 

%% Transform cell array of character vectors into double arrays 
for i = 1:size(data, 2)
    if iscell(data{:, i})  
        data.(data.Properties.VariableNames{i}) = str2double(strrep(data{:,i}, ',', '.'));
    end
end

summary(data)

%% Ensure the date column is in the datetime format
if iscell(data{:,1})
    data.date = datetime(data{:,1}, 'InputFormat', 'yyyy-MM-dd HH:mm:ss'); 
end

%% Check for missing values
missing = ismissing(data);
disp(data(any(missing, 2), :));

%% Clean up variable names
data.Properties.VariableNames = strrep(data.Properties.VariableNames, '_', '');

%% Display the corrected variable names
disp('Corrected Variable Names:');
disp(data.Properties.VariableNames');


silicaFeedColumn = 'xSilicaFeed';
silicaConcentrateColumn = 'xSilicaConcentrate'; 

% Time Series Plot of % Silica Feed and % Silica Concentrate
figure;
hold on;

% Plot using the actual column names
plot(data.date, data{:, silicaFeedColumn}, 'b', 'DisplayName', '% Silica Feed'); 
plot(data.date, data{:, silicaConcentrateColumn}, 'r', 'DisplayName', '% Silica Concentrate');

xlabel('Time');
ylabel('Silica Concentration (%)');
title('Silica Content Trend Over Time');
legend show;
hold off;

%% Correlation Heatmap
% Calculate correlation matrix (excluding date)
numeric_data = data{:, 2:end};  % Exclude the first column 
corr_matrix = corr(numeric_data, 'Rows', 'complete');

% Plot heatmap
figure;
heatmap(data.Properties.VariableNames(2:end), data.Properties.VariableNames(2:end), corr_matrix, 'Colormap', jet, 'ColorbarVisible', 'on');
title('Correlation Heatmap of Mining Process Variables');

%% Box Plot for Variables
figure;
for i = 2:size(data, 2)  % Skip the 'date' column
    subplot(5, 5, i-1);  
    boxplot(data{:, i});
    title(data.Properties.VariableNames{i});
    ylabel(data.Properties.VariableNames{i});
end
sgtitle('Box Plot of Variables'); 
%% Histograms of Variables
figure;
for i = 2:size(data, 2)  % Skip the 'date' column
    subplot(5, 5, i-1);  % Adjust the number of subplots based on the number of variables
    histogram(data{:, i});
    title(['Histogram of ', data.Properties.VariableNames{i}]);
    ylabel('Frequency');
end
sgtitle('Histograms of Variables'); 
