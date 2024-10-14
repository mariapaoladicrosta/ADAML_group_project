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

for m = [12, 24]  % number of samples in a batch

    nseg = length(X_Cal(:,1)) - (m + 1);

    for j = 1:10 % number of latent variables 

        for i = 1:nseg % number of segments of m samples

            % Center and scale the X-matrices

             % Calibration
            [XTrain, mu, sig]    = zscore(X_Cal(i:(i+m),:));
            
            % Validation
            XVal                 = normalize(X_Cal((i+m+1),:), 'Center', mu, 'scale', sig);


            % Center the Y-matrices

            % Calibration
            yTrain       = y_Cal(i:(i+m),:) - mean(y_Cal(i:(i+m),:));
            
            % Validation
            yVal      = y_Cal((i+m+1),:) - mean(y_Cal(i:(i+m),:));
            

            % Computing PLS model
            [~, ~, ~, ~, model(m).ncomp(j).segment(i).B, ...
                model(m).ncomp(j).segment(i).MSE, ...
                model(m).ncomp(j).segment(i).stats] = plsregress(XTrain, yTrain, j);
            
           

            % Calculate R2
            yfitPLS_train = [ones(height(XTrain), 1), XTrain] * model(m).ncomp(j).segment(i).B;
            TSSRes  = sum((yTrain - mean(yTrain)).^2);
            RSSRes  = sum((yTrain - yfitPLS_train).^2);
            model(m).ncomp(j).R2(i) = 1 - RSSRes / TSSRes;

            % calculate Q2
            yfitPLS_val = [ones(height(XVal), 1), XVal] * model(m).ncomp(j).segment(i).B; 
            PRESS = sum((yVal - yfitPLS_val).^2);
            model(m).ncomp(j).Q2(i) = 1 - PRESS / TSSRes;

            % RMSE
            model(m).ncomp(j).MSE(i) = sqrt(mean((yfitPLS_val - yVal).^2));

            % Storing for later
            model(m).ncomp(j).B(i,:) = model(m).ncomp(j).segment(i).B;
            
        end
        model(m).ncomp(j).Q2(isnan(model(m).ncomp(j).Q2))   = 0;
        model(m).ncomp(j).Q2(find(model(m).ncomp(j).Q2==-Inf))  = 0;
        model(m).ncomp(j).meanR2 = nanmean(model(m).ncomp(j).R2');
        model(m).ncomp(j).meanQ2 = nanmean(model(m).ncomp(j).Q2');

        model(m).ncomp(j).MSE(isnan(model(m).ncomp(j).MSE))   = 0;
        model(m).ncomp(j).MSE(find(model(m).ncomp(j).MSE ==-Inf))  = 0;
        model(m).ncomp(j).meanMSE = nanmean(model(m).ncomp(j).MSE');
    end
end


 m = [12, 24];

 for i = 1:length(m)
     for j = 1:10
         R2(i,j) = model(m(i)).ncomp(j).meanR2;
         Q2(i,j) = model(m(i)).ncomp(j).meanQ2;
         MSE(i,j) = model(m(i)).ncomp(j).meanMSE;
     end
 end


 figure;
 subplot(1,2,1)
 yvalues = {'12','24'};
 xvalues = {'1', '2', '3', '4', '5', '6', '7', '8','9','10'};
 heatmap(xvalues, yvalues, R2);
 ylabel("Observations in window frame");
 xlabel("No. components in the model");
 title("R2 values")

 subplot(1,2,2)
 yvalues = {'12','24'};
 xvalues = {'1', '2', '3', '4', '5', '6', '7', '8','9','10'};
 heatmap(xvalues, yvalues, Q2);
 ylabel("Observations in window frame");
 xlabel("No. components in the model");
 title("Q2 values");

 figure;
 yvalues = {'12','24'};
 xvalues = {'1', '2', '3', '4', '5', '6', '7', '8','9','10'};
 heatmap(xvalues, yvalues, MSE);
 ylabel("Observations in window frame");
 xlabel("No. components in the model");
 title("MSE values");

 %% Check NaN in betas
 
 for m = [12,24]
    for j = 1:10
        if any(isnan(model(m).ncomp(j).B))
            fprintf('NaN values found in betas window (%d) LV(%d)\n', m, j)
        end
    end
end

 %% Delete rows with more than 20 equal consecutive values
delete_rows = false(height(selected_data), 1);
consecutive_threshold = 20;

for var_idx = 3:25

    variable_data = selected_data{:, var_idx};
    
    start_idx = 1;
    while start_idx <= length(variable_data)
        end_idx = start_idx;
        
        % check consecutive values
        while end_idx < length(variable_data) && variable_data(end_idx) == variable_data(end_idx + 1)
            end_idx = end_idx + 1;
        end
        
        % check if the consecutive period meets the threshold
        consecutive_hours = end_idx - start_idx + 1;
        if consecutive_hours > consecutive_threshold
            % mark the rows for deletion within this constant period
            delete_rows(start_idx:end_idx) = true;
        end

        start_idx = end_idx + 1;
    end
end

% delete the marked rows from 'selected data'
selected_data_filtered = selected_data(~delete_rows, :);

fprintf('Deleted %d rows with constant periods over the threshold.\n', sum(delete_rows));

%% new data splitting
clear num_obs partition Calibration Test X_Cal y_Cal X_test y_test X_Cal_scaled y_Cal_centered

num_obs = height(selected_data_filtered);
partition = tspartition(num_obs ,"Holdout",0.2);

Calibration = selected_data_filtered(training(partition), :);
Test = selected_data_filtered(test(partition), :);


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

%% Repeat Cross-Validation
c lear model

for m = [22, 24, 28, 32, 35]  % number of samples in a batch

    nseg = length(X_Cal(:,1)) - (m + 2);

    for j = 1:10 % number of latent variables 

        for i = 1:nseg % number of segments of m samples

            % Center and scale the X-matrices

             % Calibration
            [XTrain, mu, sig]    = zscore(X_Cal(i:(i+m),:));
            
            % Validation
            XVal                 = normalize(X_Cal((i+m+1):(i+m+2),:), 'Center', mu, 'scale', sig);


            % Center the Y-matrices

            % Calibration
            yTrain       = y_Cal(i:(i+m),:) - mean(y_Cal(i:(i+m),:));
            
            % Validation
            yVal      = y_Cal((i+m+1):(i+m+2),:) - mean(y_Cal(i:(i+m),:));
            

            % Computing PLS model
            [~, ~, ~, ~, model(m).ncomp(j).segment(i).B, ...
                model(m).ncomp(j).segment(i).MSE, ...
                model(m).ncomp(j).segment(i).stats] = plsregress(XTrain, yTrain, j);


            % Calculate R2
            yfitPLS_train = [ones(height(XTrain), 1), XTrain] * model(m).ncomp(j).segment(i).B;
            TSSRes  = sum((yTrain - mean(yTrain)).^2);
            RSSRes  = sum((yTrain - yfitPLS_train).^2);
            model(m).ncomp(j).R2(i) = 1 - RSSRes / TSSRes;

            % calculate Q2
            yfitPLS_val = [ones(height(XVal), 1), XVal] * model(m).ncomp(j).segment(i).B;  % Predict on the validation set
            PRESS = sum((yVal - yfitPLS_val).^2);
            model(m).ncomp(j).Q2(i) = 1 - PRESS / TSSRes;

            % Storing for later
            model(m).ncomp(j).B(i,:) = model(m).ncomp(j).segment(i).B';

            %model(m).ncomp(j).mse_values(i) = model(m).ncomp(j).segment(i).MSE;
            
        end
        model(m).ncomp(j).Q2(isnan(model(m).ncomp(j).Q2))   = 0;
        model(m).ncomp(j).Q2(find(model(m).ncomp(j).Q2==-Inf))  = 0;
        model(m).ncomp(j).meanR2 = nanmean(model(m).ncomp(j).R2');
        model(m).ncomp(j).meanQ2 = nanmean(model(m).ncomp(j).Q2'); 
        %model(m).ncomp(j).meanMSE = mean(mse_values);
        
    end
end


m = [22, 24, 28, 32, 35];

 for i = 1:length(m)
     for j = 1:10
         R2(i,j) = model(m(i)).ncomp(j).meanR2;
         Q2(i,j) = model(m(i)).ncomp(j).meanQ2;
     end
 end

 figure;
 subplot(1,2,1)
 yvalues = {'22','24','28','32','35'};
 xvalues = {'1', '2', '3', '4', '5', '6', '7', '8','9','10'};
 heatmap(xvalues, yvalues, R2);
 ylabel("Observations in window frame");
 xlabel("No. components in the model");
 title("R2 values")

 subplot(1,2,2)
 yvalues = {'22','24','28','32','35'};
 xvalues = {'1', '2', '3', '4', '5', '6', '7', '8','9','10'};
 heatmap(xvalues, yvalues, Q2);
 ylabel("Observations in window frame");
 xlabel("No. components in the model");
 title("Q2 values");

%% Plot Q2 over time

for m = [22, 24, 28, 32, 35]
    figure;
    hold on;
    for j = 1:10
        % extract Q2 values for the current m and j
        Q2_values = model(m).ncomp(j).Q2;
        
        % plot Q2 over segments for this latent variable
        plot(Q2_values, 'DisplayName', ['LV ' num2str(j)]);
    end
    title(['Q2 over segments for m = ' num2str(m)]);
    xlabel('Segment');
    ylabel('Q2');
    legend('show');
    grid on;
    hold off;
end

%% Outliers Removal 
X = selected_data_filtered{:, 3:25};

[coeff, score, latent, ~, explained] = pca(X, 'Centered', true);

num_components = find((cumsum(explained)) / sum(explained) >= 0.95, 1);

T2_reduced = sum((score(:, 1:num_components) ./ sqrt(latent(1:num_components)')) .^ 2, 2);

mean_T2 = mean(T2_reduced);  
std_T2 = std(T2_reduced); 
control_limit_T2_2std = mean_T2 + 2 * std_T2;  
control_limit_T2_3std = mean_T2 + 3 * std_T2;

% identify outliers that exceed the 3 standard deviations control limit
outliers_T2 = find(T2_reduced > control_limit_T2_3std);

figure;
plot(T2_reduced, '-o');  
hold on;
yline(control_limit_T2_2std, '--r', '2 Std Dev');
yline(control_limit_T2_3std, '--g', '3 Std Dev');
plot(outliers_T2, T2_reduced(outliers_T2), 'ro'); % Highlight outliers
title('T² Control Chart');
xlabel('Sample');
ylabel('T²');
legend('T²', '2 Std Dev', '3 Std Dev', 'Outliers');
grid on;
hold off;


selected_data_filtered_no_outliers = selected_data_filtered;
selected_data_filtered_no_outliers(outliers_T2, :) = [];

%% New data splitting
clear num_obs partition Calibration Test X_Cal y_Cal X_test y_test X_Cal_scaled y_Cal_centered

num_obs = height(selected_data_filtered);
partition = tspartition(num_obs ,"Holdout",0.2);

Calibration = selected_data_filtered(training(partition), :);
Test = selected_data_filtered(test(partition), :);


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

%% Repeat Cross-Validation
 clear model

for m = [22, 24, 28, 32, 35]  % number of samples in a batch

    nseg = length(X_Cal(:,1)) - (m + 2);

    for j = 1:10 % number of latent variables 

        for i = 1:nseg % number of segments of m samples

            % Center and scale the X-matrices

             % Calibration
            [XTrain, mu, sig]    = zscore(X_Cal(i:(i+m),:));
            
            % Validation
            XVal                 = normalize(X_Cal((i+m+1):(i+m+2),:), 'Center', mu, 'scale', sig);


            % Center the Y-matrices

            % Calibration
            yTrain       = y_Cal(i:(i+m),:) - mean(y_Cal(i:(i+m),:));
            
            % Validation
            yVal      = y_Cal((i+m+1):(i+m+2),:) - mean(y_Cal(i:(i+m),:));
            

            % Computing PLS model
            [~, ~, ~, ~, model(m).ncomp(j).segment(i).B, ...
                model(m).ncomp(j).segment(i).MSE, ...
                model(m).ncomp(j).segment(i).stats] = plsregress(XTrain, yTrain, j);


            % Calculate R2
            yfitPLS_train = [ones(height(XTrain), 1), XTrain] * model(m).ncomp(j).segment(i).B;
            TSSRes  = sum((yTrain - mean(yTrain)).^2);
            RSSRes  = sum((yTrain - yfitPLS_train).^2);
            model(m).ncomp(j).R2(i) = 1 - RSSRes / TSSRes;

            % calculate Q2
            yfitPLS_val = [ones(height(XVal), 1), XVal] * model(m).ncomp(j).segment(i).B;  % Predict on the validation set
            PRESS = sum((yVal - yfitPLS_val).^2);
            model(m).ncomp(j).Q2(i) = 1 - PRESS / TSSRes;

            % Storing for later
            model(m).ncomp(j).B(i,:) = model(m).ncomp(j).segment(i).B';

            %model(m).ncomp(j).mse_values(i) = model(m).ncomp(j).segment(i).MSE;
            
        end
        model(m).ncomp(j).Q2(isnan(model(m).ncomp(j).Q2))   = 0;
        model(m).ncomp(j).Q2(find(model(m).ncomp(j).Q2==-Inf))  = 0;
        model(m).ncomp(j).meanR2 = nanmean(model(m).ncomp(j).R2');
        model(m).ncomp(j).meanQ2 = nanmean(model(m).ncomp(j).Q2'); 
        %model(m).ncomp(j).meanMSE = mean(mse_values);
        
    end
end


m = [22, 24, 28, 32, 35];

 for i = 1:length(m)
     for j = 1:10
         R2(i,j) = model(m(i)).ncomp(j).meanR2;
         Q2(i,j) = model(m(i)).ncomp(j).meanQ2;
     end
 end

 figure;
 subplot(1,2,1)
 yvalues = {'22','24','28','32','35'};
 xvalues = {'1', '2', '3', '4', '5', '6', '7', '8','9','10'};
 heatmap(xvalues, yvalues, R2);
 ylabel("Observations in window frame");
 xlabel("No. components in the model");
 title("R2 values")

 subplot(1,2,2)
 yvalues = {'22','24','28','32','35'};
 xvalues = {'1', '2', '3', '4', '5', '6', '7', '8','9','10'};
 heatmap(xvalues, yvalues, Q2);
 ylabel("Observations in window frame");
 xlabel("No. components in the model");
 title("Q2 values");

%% Betas Barplot

varImp = nanmean(abs(model(28).ncomp(5).B));

X_varNames = selected_data.Properties.VariableNames(3:end);

figure;
bar(varImp(2:end));
xtickangle(45);
xticklabels(gca,X_varNames);



%% Test 
[XLoadings, yLoadings, XScore, yScore, betaPLS, PLSVar] = plsregress(X_Cal_scaled, y_Cal_centered, 3);
yfitPLS = [ones(height(X_test_scaled), 1), X_test_scaled]*betaPLS;

TSS = sum((y_test_centered - mean(y_test_centered)).^2);
RSSPLS = sum((y_test_centered - yfitPLS).^2);
RquaredPLS = 1 - RSSPLS/TSS





