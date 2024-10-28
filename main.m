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

%% printable table for statistics
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
    boxplot(functioning{:, i});
    title(['Histogram of ', functioning.Properties.VariableNames{i}]);
    ylabel('Frequency');
end
sgtitle('Histograms of Variables'); 

%% Resampling 

resampled = resampleByStdThreshold(functioning);

%% Add Lags

lag_vars = resampled.Properties.VariableNames(2:end);
num_lags = 6;
lagged_data = LaggingBySegments(resampled, lag_vars, num_lags, 'xSilicaConcentrate');

%% Selection based on R2
% x_cols = [25, 162];
% y_col = 163;
% PLS_components = 4;
% 
% selected_data = R2PLSVariables(lagged_data, x_cols, y_col, PLS_components);

%% Selection based on correlation
x_cols = [25, 162];
y_col = 163;

selected_data = HighestCorrelationVariables(lagged_data, x_cols, y_col, num_lags);

%% Delete rows with more than 20 equal consecutive values
consecutive_threshold = 20;
selected_data = deleteConstantPeriods(selected_data, consecutive_threshold);

%% T2 and SPEx control charts
X = selected_data{:,3:end};
explained_variance_threshold = 0.95;
[T2_reduced, SPEx, outliers_T2, outliers_SPEx, control_limits_T2,control_limits_SPEx] = computeT2_SPEx(X, explained_variance_threshold);


figure;

% Plot T² Chart
subplot(2,1,1)
hold on;
plot(T2_reduced, '-o', 'DisplayName', 'T²');
yline(control_limits_T2(1), '--r', 'T² 2 Std Dev', 'DisplayName', 'Warning Limit');
yline(control_limits_T2(2), '--g', 'T² 3 Std Dev', 'DisplayName', 'Alarm Limit');
plot(outliers_T2, T2_reduced(outliers_T2), 'ro', 'DisplayName', 'T² Outliers');
title('T² Control Chart');
xlabel('Sample');
ylabel('T² Value');
legend('Location', 'best');
grid on;
hold off;

% Plot SPEx Chart
subplot(2,1,2)
hold on;
plot(SPEx, '-o', 'DisplayName', 'SPEx');
yline(control_limits_SPEx(1), '--b', 'SPEx 2 Std Dev', 'DisplayName', 'Warning Limit');
yline(control_limits_SPEx(2), '--m', 'SPEx 3 Std Dev', 'DisplayName', 'Alarm Limit');
plot(outliers_SPEx, SPEx(outliers_SPEx), 'mo', 'DisplayName', 'SPEx Outliers'); % Highlight SPEx outliers
title('SPEx Control Chart');
xlabel('Sample');
ylabel('SPEx Value');
legend('Location', 'best');
grid on;
hold off;

% common outliers
common_outliers = intersect(outliers_T2, outliers_SPEx);


disp('Common Outliers in T² and SPEx charts:');
disp(common_outliers);

%% T2 and SPEx Contributions

X_standardized = (X - mean(X)) ./ std(X);
[coeff, score, latent, ~, explained] = pca(X_standardized);
num_components = find((cumsum(explained)) / sum(explained) >= explained_variance_threshold, 1);

for i = 1:length(common_outliers)
    outlier_index = common_outliers(i);
    
    % data of current outlier
    outlier_data = X_standardized(outlier_index, :);
    
    % T² contributions for the current outlier using t2contr
    T2_contributions = t2contr(outlier_data, coeff, latent, num_components);
    
    % SPEx contributions for the current outlier using qcontr
    SPEx_contributions = qcontr(outlier_data, coeff, num_components);
    
    % T² contributions for the current outlier
    figure;
    subplot(2, 1, 1);
    bar(T2_contributions);
    title(['T² Contributions for Outlier ', num2str(outlier_index)]);
    xlabel('Variable');
    ylabel('T² Contribution');
    grid on;
    
    % SPEx contributions for the current outlier
    subplot(2, 1, 2);
    bar(SPEx_contributions);
    title(['SPEx Contributions for Outlier ', num2str(outlier_index)]);
    xlabel('Variable');
    ylabel('SPEx Contribution');
    grid on;
end

%% Remove outliers in common
selected_data_filtered = selected_data;
selected_data_filtered(common_outliers, :) = [];

%% Data Splitting & Scaling
holdout_ratio = 0.2;
[Calibration, Test, X_Cal, y_Cal, X_Cal_scaled, y_Cal_centered, X_test_scaled, y_test_centered] = SplittingScaling(selected_data_filtered, holdout_ratio);

% visualization
figure;
for i = 2:23 
    subplot(5, 5, i-1); 
    histogram(X_Cal_scaled(:, i));
    title(['Histogram of ', Calibration.Properties.VariableNames{i+2}]);
    ylabel('Frequency');
end
sgtitle('Histograms of Variables'); 

%% Cross Validation through Sliding Windows
m = [22, 24, 28, 32, 35, 40, 45];
max_components = 10;
model = CrossValidation_SlidingWindow(X_Cal, y_Cal, m, max_components);

%% R2 and Q2 values for each Window size and nr of LVs

 for i = 1:length(m)
     for j = 1:10
         R2(i,j) = model(m(i)).ncomp(j).meanR2;
         Q2(i,j) = model(m(i)).ncomp(j).meanQ2;
     end
 end

 figure;
 subplot(1,2,1)
 yvalues = {'22','24','28','32','35','40', '45'};
 xvalues = {'1', '2', '3', '4', '5', '6', '7', '8','9','10'};
 heatmap(xvalues, yvalues, R2);
 ylabel("Observations in window frame");
 xlabel("No. components in the model");
 title("R2 values")

 subplot(1,2,2)
 yvalues = {'22','24','28','32','35','40', '45'};
 xvalues = {'1', '2', '3', '4', '5', '6', '7', '8','9','10'};
 heatmap(xvalues, yvalues, Q2);
 ylabel("Observations in window frame");
 xlabel("No. components in the model");
 title("Q2 values");

 %%  Q2 Analysis
[model, avg_positive_Q2] = Q2ValuesAnalysis(Calibration, model, m, max_components);

% heatmap of positive Q2
figure;
yvalues = {'22','24','28','32','35','40', '45'};
xvalues = {'1', '2', '3', '4', '5', '6', '7', '8','9','10'};
heatmap(xvalues, yvalues, avg_positive_Q2);

title('Average Positive Q² Values Heatmap');
xlabel('Number of Latent Variables (Components)');
ylabel('Window Size (m)');
colorbar;

 
 %% Plot Q2 over time
configurations = [24, 3; 32, 7]; 

figure;
for k = 1:size(configurations, 1)

    m = configurations(k, 1);
    j = configurations(k, 2);

    Q2_values = model(m).ncomp(j).Q2;
    
    subplot(1,2,k)
    plot(Q2_values, '-o');
    title(['Q2 over segments for m = ' num2str(m) ', with LVs = ' num2str(j)]);
    xlabel('Segment');
    ylabel('Q2');
    grid on;
    hold off;
end

%% Residuals over time
configurations = [24, 3; 32, 7]; 

figure;
for k = 1:size(configurations, 1)
    m = configurations(k, 1);
    j = configurations(k, 2);
    
    subplot(1, 2, k);
    plot(model(m).ncomp(j).residuals, '-o', 'DisplayName', ['LV ' num2str(j)]);
    xlabel('Segment');
    ylabel('Residuals');
    title(['Residuals over time for m = ' num2str(m) ', LVs = ' num2str(j)]);
    legend show;
    grid on;
end

%% Negative Q2 analysis
% dates for each configuration
dates_1 = model(24).ncomp(3).negative_Q2.Dates;
dates_2 = model(32).ncomp(7).negative_Q2.Dates;

% common dates between the two configurations
common_dates = intersect(dates_1, dates_2);

disp('Common dates between the two configurations:');
disp(common_dates);

% month and weekday information
months = month(common_dates); % Month as a number (1 to 12)
weekdays = weekday(common_dates); % Weekday as a number (1 = Sunday, 7 = Saturday)

% months (April to August)
selected_months = 4:8;


occurrences_by_weekday_month = zeros(7, length(selected_months));

% counts for each day of the week within selected months
for m_idx = 1:length(selected_months)
    month_num = selected_months(m_idx);
    for d = 1:7
        occurrences_by_weekday_month(d, m_idx) = sum(weekdays == d & months == month_num);
    end
end


figure;
subplot(2, 1, 1);
month_counts = histcounts(months, 1:13); % Count occurrences by month
bar(1:12, month_counts, 'FaceColor', [0.2, 0.6, 0.8]);
xticks(1:12);
xticklabels({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'});
xlabel('Month');
ylabel('Occurrences');
title('Occurrences of Common Dates by Month');
grid on;


subplot(2, 1, 2);
bar(occurrences_by_weekday_month, 'stacked');
colormap(jet(length(selected_months))); % Set color map for differentiation
xticks(1:7);
xticklabels({'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'});
xlabel('Day of the Week');
ylabel('Occurrences');
title('Occurrences of Common Dates by Day of the Week (April to August)');
legend({'April', 'May', 'June', 'July', 'August'}, 'Location', 'best');
grid on;

%% beta over time

figure;
for i = 1:size(configurations, 1)
    m_value = configurations(i, 1);
    lv = configurations(i, 2);

    num_segments = length(model(m_value).ncomp(lv).segment);
    num_vars = length(model(m_value).ncomp(lv).segment(1).B);
    coeff = zeros(num_vars, num_segments);

    for k = 1:num_segments
        coeff(:, k) = model(m_value).ncomp(lv).segment(k).B;
    end


    subplot(1,2,i)
    heatmap(coeff(:, 1:200));
    title(sprintf('PLS Coefficients for Window Size %d, %d Latent Variables', m_value, lv));
    xlabel('Segment');
    ylabel('Variable');
    colorbar;
end

%% Avg beta

X_varNames = Calibration.Properties.VariableNames(3:end); 

figure;

for i = 1:size(configurations, 1)
    m_value = configurations(i, 1);
    lv = configurations(i, 2);

    varImp = mean(abs(model(m_value).ncomp(lv).B));
    
    subplot(2,1,i)
    bar(varImp(2:end));  % Exclude intercept (if applicable)
    set(gca, 'XTick', 1:numel(X_varNames), 'XTickLabel', X_varNames);
    xtickangle(45);
    title(sprintf('Variable Importance for Window Size %d, %d Latent Variables', m_value, lv));
    ylabel('Mean Absolute Beta Coefficient');
    xlabel('Predictor Variables');
    grid on;
end


%% Test
[XLoadings, yLoadings, XScore, yScore, betaPLS, PLSVar] = plsregress(X_Cal_scaled, y_Cal_centered, 3);
yfitPLS = [ones(height(X_test_scaled), 1), X_test_scaled]*betaPLS;

TSS = sum((y_test_centered - mean(y_test_centered)).^2);
RSSPLS = sum((y_test_centered - yfitPLS).^2);
RquaredPLS = 1 - RSSPLS/TSS

%% VIP
X_varNames = Calibration.Properties.VariableNames(3:end);
VIP = computeVIP(XLoadings, yScore, 3, TSS);

figure;
bar(VIP);
xlabel('Predictor Variables');
ylabel('VIP Score');
set(gca, 'XTick', 1:numel(X_varNames), 'XTickLabel', X_varNames);
title('Variable Importance in Projection (VIP) Scores');
grid on;

%%
% select only the columns with high VIP
high_VIP_variables = find(VIP > 2500);

X_Cal_reduced = X_Cal_scaled(:, high_VIP_variables);
X_test_reduced = X_test_scaled(:, high_VIP_variables);


numComponents = 4; 
[XLoadings, yLoadings, XScore, yScore, betaPLS, PLSVar] = plsregress(X_Cal_reduced, y_Cal_centered, numComponents);

% prediction with recalibrated model
yfitPLS = [ones(height(X_test_reduced), 1), X_test_reduced] * betaPLS;

% R² for the recalibrated model
TSS = sum((y_test_centered - mean(y_test_centered)).^2);
RSSPLS = sum((y_test_centered - yfitPLS).^2);
RquaredPLS = 1 - RSSPLS / TSS;

disp('Recalibrated R²:');
disp(RquaredPLS);

%% betas analysis
X_varNames = Calibration.Properties.VariableNames(2+high_VIP_variables);

figure;
bar(betaPLS(2:end));
set(gca, 'XTick', 1:numel(X_varNames), 'XTickLabel', X_varNames);

%%
% Calculate residuals
residualsPLS = abs(y_test_centered - yfitPLS);

% Plot the residuals
figure;
plot(residualsPLS, 'o-', 'MarkerSize', 6, 'LineWidth', 1.5);
title('Residuals of PLS Regression');
xlabel('Observation');
ylabel('Residuals');
grid on;

%%
high_residuals_idx = find(residualsPLS > 2);

% dates corresponding to these high residuals
dates_high_residuals = Test{high_residuals_idx,1};
