clear all;
close all;
clc;

% loading the file
data = readtable('MiningProcess_Flotation_Plant_Database.csv');

%% Exploring data
whos data

size(data)

head(data)

data.Properties.VariableNames = strrep(data.Properties.VariableNames, '_', '');
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

% Time Series Plot of % Iron and Silica Feed and % Iron and Silica Concentrate
figure;
hold on;

% Plot using the actual column names
plot(data.date, data{:, 'xSilicaFeed'}, 'cyan', 'DisplayName', '% Silica Feed'); 
plot(data.date, data{:, 'xIronFeed'}, 'magenta', 'DisplayName', '% Silica Feed'); 
plot(data.date, data{:, 'xIronConcentrate'}, 'b', 'DisplayName', '% Iron Concentrate');
plot(data.date, data{:, 'xSilicaConcentrate'}, 'r', 'DisplayName', '% Silica Concentrate');

xlabel('Time');
ylabel('Silica Concentration (%)');
title('Silica Content Trend Over Time');
legend show;
hold off;

% select functioning period
functioning = data(data.date >= datetime(2017, 3, 29, 21, 0, 0), :);

%% Correlation Heatmap
% Calculate correlation matrix (excluding date)
numeric_data = functioning{:, 2:end};  
corr_matrix = corr(numeric_data, 'Rows', 'complete');

% Plot heatmap
figure;
heatmap(functioning.Properties.VariableNames(2:end), functioning.Properties.VariableNames(2:end), corr_matrix, 'Colormap', jet, 'ColorbarVisible', 'on');
title('Correlation Heatmap of Mining Process Variables');

%% Histograms of Variables
figure;
for i = 2:size(functioning, 2)  
    subplot(5, 5, i-1); 
    histogram(functioning{:, i});
    title(['Histogram of ', functioning.Properties.VariableNames{i}]);
    ylabel('Frequency');
end
sgtitle('Histograms of Variables'); 

%% Resampling
[group_idx, grouped_dates] = findgroups(functioning.date);
std_table = table(grouped_dates, 'VariableNames', {'date'});

for i = 2:width(functioning)
    % apply the std function to each group date-time
    std_per_hour = splitapply(@std, functioning{:, i}, group_idx);
    std_table.(functioning.Properties.VariableNames{i}) = std_per_hour;
end

thresholds = varfun(@mean, std_table, 'InputVariables', std_table.Properties.VariableNames(2:end));

resampled = table(grouped_dates, 'VariableNames', {'date'});

for i = 2:width(functioning)

    resampled_values = zeros(max(group_idx), 1); 
    % threshold for the current variable
    threshold = thresholds.(thresholds.Properties.VariableNames{i-1});
    
    % loop through each group date-time
    for group = 1:max(group_idx)
        group_data = functioning{group_idx == group, i};
        
        % standard deviation for this group
        group_std = std(group_data);
        
        % resample based on the standard deviation
        if group_std > threshold
            % if std is greater than the threshold, use median
            resampled_values(group) = median(group_data);
        else
            % if std is less than or equal to the threshold, use mean
            resampled_values(group) = mean(group_data);
        end
    end
    % add the resampled values as a new column in the resampled table
    resampled.(functioning.Properties.VariableNames{i}) = resampled_values;
end

%% Lagged Variables
lagged_data = resampled;

% loop through each variable
for i = 2:width(resampled)

    % lagged variables for t-1, t-2, t-3
    lagged_data.(['Lag1_', resampled.Properties.VariableNames{i}]) = lagmatrix(resampled{:,i}, 1);  % t-1 lag
    lagged_data.(['Lag2_', resampled.Properties.VariableNames{i}]) = lagmatrix(resampled{:,i}, 2);  % t-2 lag
    lagged_data.(['Lag3_', resampled.Properties.VariableNames{i}]) = lagmatrix(resampled{:,i}, 3);  % t-3 lag
end

% eliminate the first 3 rows, which contain NaN values 
lagged_data = lagged_data(4:end, :);

% Heatmap
original_vars = data.Properties.VariableNames(2:end)

% Create an ordered list of x-axis variables: each variable followed by its lags
x_vars_ordered = {};
for i = 1:length(original_vars)
    x_vars_ordered = [x_vars_ordered, original_vars{i}, ...
                      ['Lag1_', original_vars{i}], ...
                      ['Lag2_', original_vars{i}], ...
                      ['Lag3_', original_vars{i}]];
end

% define y-axis variables (Silica Concentrate and its lags)
y_vars = {'xSilicaConcentrate', 'Lag1_xSilicaConcentrate', 'Lag2_xSilicaConcentrate', 'Lag3_xSilicaConcentrate'};

% remove y-axis variables from x-axis variables (so they only appear on the y-axis)
x_vars = setdiff(x_vars_ordered, y_vars, 'stable');  % 'stable' keeps the original order

% extract numeric data for the x-axis (variables + lags) and y-axis (% Silica Concentrate + lags)
x_numeric_data = lagged_data{:, x_vars};  % All other variables
y_numeric_data = lagged_data{:, y_vars};  % Silica Concentrate and its lags

% correlation matrix between y-axis and x-axis variables
corr_matrix = corr(y_numeric_data, x_numeric_data, 'Rows', 'complete');

% plot heatmap
figure;
h = heatmap(x_vars, y_vars, corr_matrix);
h.Colormap = jet;
h.ColorbarVisible = 'on';
title('Correlation Heatmap: % Silica Concentrate and Lags vs Other Variables (Grouped with Lags)');

%% Data Splitting 
% ratios
calibration_ratio = 0.6;
validation_ratio = 0.2;
test_ratio = 0.2;

num_obs = height(lagged_data);
calibration_idx = 1:round(calibration_ratio * num_obs);
validation_idx = (round(calibration_ratio * num_obs) + 1):round((calibration_ratio + validation_ratio) * num_obs);
test_idx = (round((calibration_ratio + validation_ratio) * num_obs) + 1):num_obs;

calibration_data = lagged_data(calibration_idx, :);
validation_data = lagged_data(validation_idx, :);
test_data = lagged_data(test_idx, :);

%% Scaling
calibration_numeric = calibration_data{:, 2:end};

% mean and standard deviation of the calibration set
calibration_mean = mean(calibration_numeric);
calibration_std = std(calibration_numeric);

% scale calibration data using its own mean and std
calibration_data_scaled = calibration_data;
calibration_data_scaled{:, 2:end} = (calibration_numeric - calibration_mean) ./ calibration_std;

% scale validation and test data using the calibration mean and std
validation_data_scaled = validation_data;
validation_data_scaled{:, 2:end} = (validation_data{:, 2:end} - calibration_mean) ./ calibration_std;

test_data_scaled = test_data;
test_data_scaled{:, 2:end} = (test_data{:, 2:end} - calibration_mean) ./ calibration_std;

% visualization
figure;
for i = 2:24  
    subplot(5, 5, i-1);
    histogram(calibration_data_scaled{:, i});
    %plot(calibration_data_scaled.date,calibration_data_scaled{:, i});
    title(['Histogram of ', calibration_data_scaled.Properties.VariableNames{i}]);
    ylabel('Frequency');
end
sgtitle('Histograms of Variables'); 


