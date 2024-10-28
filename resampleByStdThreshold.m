function resampled = resampleByStdThreshold(data)
    % This function resamples data based on the standard deviation threshold.
    % If the standard deviation per hour exceeds the threshold, the median is used.
    % Otherwise, the mean is used.
    %
    % Parameters:
    %   data - A table where the first column is a date variable, and the
    %          remaining columns are numeric variables to be resampled.
    %
    % Returns:
    %   resampled - A table with resampled data based on the standard deviation threshold.

    % group data by date
    [group_idx, grouped_dates] = findgroups(data.date);
    std_table = table(grouped_dates, 'VariableNames', {'date'});

    % standard deviation per group for each variable
    for i = 2:width(data)
        std_per_hour = splitapply(@std, data{:, i}, group_idx);
        std_table.(data.Properties.VariableNames{i}) = std_per_hour;
    end

    % mean threshold for each variable
    thresholds = varfun(@mean, std_table, 'InputVariables', std_table.Properties.VariableNames(2:end));

    resampled = table(grouped_dates, 'VariableNames', {'date'});

    % resample each variable based on standard deviation threshold
    for i = 2:width(data)
        resampled_values = zeros(max(group_idx), 1); 
        threshold = thresholds.(thresholds.Properties.VariableNames{i-1});

        % loop through each group
        for group = 1:max(group_idx)
            group_data = data{group_idx == group, i};
            group_std = std(group_data);

            % resampling based on threshold
            if group_std > threshold
                resampled_values(group) = median(group_data);
            else
                resampled_values(group) = mean(group_data);
            end
        end
        
        % add the resampled values to the resampled table
        resampled.(data.Properties.VariableNames{i}) = resampled_values;
    end
end
