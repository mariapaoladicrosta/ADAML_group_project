clear all;
close all;
clc;

% loading the file
data = readtable('MiningProcess_Flotation_Plant_Database.csv');

%% Exploring data
whos data

size(data)

head(data)

% Check the variable names to identify the correct columns
disp(data.Properties.VariableNames'); 

%% 
%Transform cell array of character vectors into double arrays 
for i = 1:size(data, 2)
    if iscell(data{:, i})  
        data.(data.Properties.VariableNames{i}) = str2double(strrep(data{:,i}, ',', '.'));
    end
end

% Ensure the date column is in the datetime format
if iscell(data{:,1})
    data.date = datetime(data{:,1}, 'InputFormat', 'yyyy-MM-dd HH:mm:ss'); 
end

summary(data)

%%
% create a printable table of summary statistics
data_summary = summary(data);
variable_names = fieldnames(data_summary);  
variable_names = variable_names(2:end);     
stats_of_interest = {'Description', 'Min', 'Median', 'Max', 'NumMissing'};


num_variables = length(variable_names);  
num_stats = length(stats_of_interest);  
data_matrix = cell(num_variables, num_stats);

for i = 1:num_variables
    for j = 1:num_stats
        stat_name = stats_of_interest{j};
        stat_value = data_summary.(variable_names{i}).(stat_name);
        data_matrix{i, j} = num2str(stat_value);
    end
end


summary_table = cell2table(data_matrix, 'RowNames', variable_names, 'VariableNames', stats_of_interest);
figure;
uitable('Data', data_matrix, ...  
        'ColumnName', stats_of_interest, ...
        'RowName', variable_names, ...
        'Position',[50 50 650 450]);
        

%% Check for missing values
missing = ismissing(data);
disp(data(any(missing, 2), :));


%%
silicaFeedColumn = 'x_SilicaFeed';
silicaConcentrateColumn = 'x_SilicaConcentrate'; 

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
numeric_data = data{:, 2:end};  
corr_matrix = corr(numeric_data, 'Rows', 'complete');

% Plot heatmap
figure;
heatmap(data.Properties.VariableNames(2:end), data.Properties.VariableNames(2:end), corr_matrix, 'Colormap', jet, 'ColorbarVisible', 'on');
title('Correlation Heatmap of Mining Process Variables');

%% Boxplots & Histograms of Variables
figure;
for i = 2:size(data, 2)  % Skip the 'date' column
    subplot(5, 5, i-1);  
    boxplot(data{:, i});
    title(data.Properties.VariableNames{i});
    ylabel(data.Properties.VariableNames{i});
end
sgtitle('Box Plot of Variables'); 

figure;
for i = 2:size(data, 2)  
    subplot(5, 5, i-1);  
    histogram(data{:, i});
    title(['Histogram of ', data.Properties.VariableNames{i}]);
    ylabel('Frequency');
end
sgtitle('Histograms of Variables'); 

