function lagged_data = LaggingBySegments(resampled, lag_vars, num_lags, lead_var)
    % Applies lagging to specific segments of a dataset based on date ranges.
    % Optionally applies a future lag to a specified variable.
    %
    % Parameters:
    %   resampled - The input table with a 'date' column and variables to be lagged.
    %   lag_vars - Cell array of variable names to be lagged.
    %   num_lags - Number of lags to apply to each variable in lag_vars.
    %   lead_var - (Optional) Name of the variable to lag into the future.
    %
    % Returns:
    %   lagged_data - Table with lagged variables for each segment and no missing values.

    if nargin < 4
        lead_var = ''; % Default to no future lag if not specified
    end

    % define the segments based on date ranges
    segments = {...
        resampled.date >= datetime(2017, 3, 30, 0, 0, 0) & resampled.date <= datetime(2017, 5, 13, 0, 0, 0), ...
        resampled.date >= datetime(2017, 6, 15, 1, 0, 0) & resampled.date <= datetime(2017, 7, 24, 0, 0, 0), ...
        resampled.date >= datetime(2017, 8, 15, 1, 0, 0)}; % Define your date ranges here
    
    lagged_data = [];

    % loop over each segment to apply lagging
    for i = 1:length(segments)
        segment_data = resampled(segments{i}, :);
        
        % store lagged data for this segment
        segment_lagged_data = segment_data;

        % apply lagging for specified variables
        for j = 1:length(lag_vars)  
            var_name = lag_vars{j};
            for lag = 1:num_lags
                if ismember(var_name, segment_data.Properties.VariableNames)
                    segment_lagged_data.(['Lag', num2str(lag), '_', var_name]) = ...
                        lagmatrix(segment_data{:, var_name}, lag);
                end
            end
        end
        
        % apply future lag for lead variable, if specified
        if ~isempty(lead_var) && ismember(lead_var, segment_data.Properties.VariableNames)
            segment_lagged_data.([lead_var, '_lead1']) = lagmatrix(segment_data{:, lead_var}, -1);
        end
        
        % append segment lagged data to the overall dataset
        lagged_data = [lagged_data; segment_lagged_data];
    end

    % remove rows with missing values introduced by the lags
    lagged_data = lagged_data(~any(ismissing(lagged_data), 2), :);
end
