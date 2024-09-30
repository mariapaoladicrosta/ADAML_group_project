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
% time Series Plot of % Iron and Silica Feed and % Iron and Silica Concentrate
figure;
hold on;

% Plot using the actual column names
plot(data.date, data{:, 'xSilicaFeed'}, 'cyan', 'DisplayName', '% Silica Feed'); 
plot(data.date, data{:, 'xIronFeed'}, 'magenta', 'DisplayName', '% Iron Feed'); 
plot(data.date, data{:, 'xIronConcentrate'}, 'b', 'DisplayName', '% Iron Concentrate');
plot(data.date, data{:, 'xSilicaConcentrate'}, 'r', 'DisplayName', '% Silica Concentrate');

xlabel('Time');
ylabel('Iron and Silica Concentration (%)');
title('Iron and Silica Content Trend Over Time');
legend show;
hold off;


% cut off the period for which we have no observations: start date 30 March
% let's start collect this 6 hours before because of Laggings (see next step)
functioning = data(data.date >= datetime(2017, 3, 29, 18, 0, 0), :);

%% Accounting for downtime periods
period1 = functioning.date >= datetime(2017, 5, 9, 0, 0, 0) & ...
          functioning.date <= datetime(2017, 6, 15, 23, 0, 0);


figure;
plot(functioning.date(period1), functioning{period1, 'xSilicaFeed'}, 'DisplayName', '% Silica Feed');
xlabel('Date');
ylabel('Silica Feed (%)');
title('Silica Feed from May 9, 2017, to June 15, 2017');
legend show;

functioning = functioning(functioning.date < datetime(2017, 5, 13, 1, 0, 0) | functioning.date > datetime(2017, 6, 15, 0, 0, 0),:);

period2 = functioning.date >= datetime(2017, 7, 23, 0, 0, 0) & ...
          functioning.date <= datetime(2017, 8, 15, 23, 0, 0);

figure;
plot(functioning.date(period2), functioning{period2, 'xSilicaFeed'}, 'DisplayName', '% Silica Feed');
xlabel('Date');
ylabel('Silica Feed (%)');
title('Silica Feed from July 23, 2017, to August 15, 2017');
legend show;

functioning = functioning(functioning.date < datetime(2017, 7, 24, 1, 0, 0) | functioning.date > datetime(2017, 8, 15, 0, 0, 0),:);

figure;
hold on;

% Plot using the actual column names
plot(functioning.date, functioning{:, 'xSilicaFeed'}, 'cyan', 'DisplayName', '% Silica Feed'); 
plot(functioning.date, functioning{:, 'xIronFeed'}, 'magenta', 'DisplayName', '% Silica Feed'); 
plot(functioning.date, functioning{:, 'xIronConcentrate'}, 'b', 'DisplayName', '% Iron Concentrate');
plot(functioning.date, functioning{:, 'xSilicaConcentrate'}, 'r', 'DisplayName', '% Silica Concentrate');

xlabel('Time');
ylabel('Silica Concentration (%)');
title('Silica Content Trend Over Time');
legend show;
hold off;


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



%% Lagged variables  (6 lags)
% time periods for continuous segments
segments = {...
    resampled.date >= datetime(2017, 3, 30, 0, 0, 0) & resampled.date <= datetime(2017, 5, 13, 0, 0, 0), ...
    resampled.date >= datetime(2017, 6, 15, 1, 0, 0) & resampled.date <= datetime(2017, 7, 24, 0, 0, 0), ...
    resampled.date >= datetime(2017, 8, 15, 1, 0, 0)}; %

lagged_data = [];

% loop over each segment to apply lagging
for i = 1:length(segments)
    % extract the continuous segment
    segment_data = resampled(segments{i}, :);
    
    % table to store lagged data for this segment
    segment_lagged_data = segment_data;
    
    % lag variables t-i for i from 1 to 6, except for xSilicaConcentrate 
    for j = 2:size(segment_data, 2)  % exclude the date column 
        for lag = 1:6
                segment_lagged_data.(['Lag', num2str(lag), '_', segment_data.Properties.VariableNames{j}]) = ...
                    lagmatrix(segment_data{:, j}, lag);
        end

    end
    
    % lag xSilicaConcentrate at t+1 (future lag)
    segment_lagged_data.xSilicaConcentrate_lead1 = lagmatrix(segment_data{:, 'xSilicaConcentrate'}, -1);
    
    lagged_data = [lagged_data; segment_lagged_data];
end

% remove rows with missing values introduced by the lags
lagged_data = lagged_data(~any(ismissing(lagged_data), 2), :);

disp('Lagged dataset:');
head(lagged_data);

% Heatmap
y_vars = {'xSilicaConcentrate_lead1'};
excluded_variables = [resampled.Properties.VariableNames, 'xSilicaConcentrate_lead1'];
x_vars = lagged_data.Properties.VariableNames(~ismember(lagged_data.Properties.VariableNames, excluded_variables));

% extract numeric data
x_numeric_data = lagged_data{:, x_vars};  % All independent variables (numeric)
y_numeric_data = lagged_data{:, y_vars};  % Silica Concentrate and its future lags (numeric)

corr_matrix = corr(y_numeric_data, x_numeric_data, 'Rows', 'complete');

% Plot heatmap
figure;
h = heatmap(x_vars, y_vars, corr_matrix);
h.Colormap = jet;
h.ColorbarVisible = 'on';
title('Correlation Heatmap: Future % Silica Concentrate vs Other Variables (Grouped with Lags)');

%% Filter the dataset based on highest correlation
selected_data = lagged_data(:, 1);

% dependent variable
selected_data.xSilicaConcentrate_lead1 = lagged_data.xSilicaConcentrate_lead1;

% indices of the lagged variables (from 25th to 156th columns)
lag_start_idx = 25;
lag_end_idx = 156;

num_variables = 23;
lags_per_variable = 6;

% loop over each group of 6 lags and select the lag with the highest correlation
for var_idx = 1:num_variables
    
    group_start_idx = (var_idx - 1) * lags_per_variable + 1;  
    group_end_idx = group_start_idx + lags_per_variable - 1; 
    
    % index of the lag with the highest correlation for the current group in the correlation matrix
    [~, max_corr_idx] = max(abs(corr_matrix(group_start_idx:group_end_idx)));
    
    % index of the selected lag in the original dataset
    selected_column_idx = lag_start_idx + (group_start_idx - 1) + (max_corr_idx - 1);
    
    % add the selected lag to the filtered data 
    selected_data = [selected_data, lagged_data(:, selected_column_idx)];
end

disp('Filtered dataset with highest correlated lags:');
head(selected_data);


%% Data Splitting & Scaling

% split into Calibration (training + validation) and Testing (model evaluation)
num_obs = height(selected_data);
partition = tspartition(num_obs ,"Holdout",0.2);

Calibration = selected_data(training(partition), :);
Test = selected_data(test(partition), :);

X_Cal = table2array(Calibration(:,3:end));
y_Cal = table2array(Calibration(:,2));

X_test = table2array(Test(:,3:end));
y_test = table2array(Test(:,2));

% standardize X calibration data
X_Cal_scaled = (X_Cal - mean(X_Cal))./ std(X_Cal);
% center y calibration data
y_Cal_centered  = y_Cal - mean(y_Cal);

% standardize X test data with training mean and std
X_test_scaled = (X_test - mean(X_Cal))./ std(X_Cal);
% center y test data
y_test_centered  = y_test - mean(y_Cal);

% visualization
figure;
for i = 2:23 
    subplot(5, 5, i-1); 
    histogram(X_Cal_scaled(:, i));
    title(['Histogram of ', Calibration.Properties.VariableNames{i+2}]);
    ylabel('Frequency');
end
sgtitle('Histograms of Variables');  


%% Calibrate PLS with Sliding Window
n = height(X_Cal);
windows = 5;
cv = tspartition(n, "SlidingWindow", windows, TrainSize = round(0.8 * n));


for i = 1:cv.NumTestSets

    trainIdx = training(cv, i);
    valIdx  = test(cv, i);

    X_train = X_Cal(trainIdx, :);
    y_train = y_Cal(trainIdx, :);
    X_val = X_Cal(valIdx, :);
    y_val = y_Cal(valIdx, :);

    % center & scale
    mu = mean(X_train); sigma = std(X_train);
    XTrain = (X_train - mu)./ sigma;
    XVal = (X_val - mu)./ sigma;
    yTrain = y_train - mean(y_train); 
    yVal = y_val - mean(y_train);
    

    TSS = sum((yTrain - mean(yTrain)).^2);

    [rows, ~] = size(XVal);

    for j = 1:10
        [XLoadings, yLoadings, XScore, yScore, betaPLS, PLSVar] = plsregress(XTrain, yTrain, j);  % Perform PLS with j latent variables
        yfitPLS = [ones(rows, 1), XVal] * betaPLS;  % Predict on the validation set
        yfitPLS_train= [ones(height(XTrain), 1), XTrain] * betaPLS;


        % performance metrics for PLS
        RMSEPLS(i,j) = sqrt(mean((yfitPLS - yVal).^2));  % Root Mean Squared Error (RMSE)

        PRESSPLS(i,j) = sum((yfitPLS - yVal).^2);  % Prediction error sum of squares (PRESS)
        Q2PLS(i,j) = 1 - (PRESSPLS(i,j) / TSS);  % Q², predictive ability

        
    end
 end

RMSE_cv = mean(RMSEPLS)
Q2_cv = mean(Q2PLS)

%%
figure;
hold on
b = plot(1:10,100*cumsum(PLSVar(1,:))/sum(PLSVar(1,:)), 'r-o'); %variance in X
c = plot(1:10,100*cumsum(PLSVar(2,:))/sum(PLSVar(2,:)), 'k-o'); %variance in Y
xlabel('Number of LVs');
ylabel('Explained Variance');
legend([b,c], {'PLS: Explained Variance in X', 'PLS: Explained Variance in Y'});


figure;
plot(Q2_cv, 'b-o');
xlabel('Number of LVs');
ylabel("Q^2_{CV}")

figure;
plot(RMSE_cv, 'b-o');
xlabel('Number of LVs');
ylabel("RMSE_{CV}");







