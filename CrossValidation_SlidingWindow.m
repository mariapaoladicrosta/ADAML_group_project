function model = CrossValidation_SlidingWindow(X_Cal, y_Cal, batch_sizes, max_components)
    % Performs PLS cross-validation on the calibration data with varying batch sizes and components.
    %
    % Parameters:
    %   X_Cal          - Calibration matrix (features).
    %   y_Cal          - Calibration vector (target variable).
    %   batch_sizes    - Array of batch sizes (number of samples per batch).
    %   max_components - Maximum number of latent variables (components) to use.
    %
    % Returns:
    %   model          - Struct containing model results, including R², Q², and residuals.

    model = struct();

    for m = batch_sizes
        nseg = length(X_Cal(:,1)) - (m + 1);

        for j = 1:max_components
            for i = 1:nseg
                % Center and scale the X matrices for calibration
                [XTrain, mu, sig] = zscore(X_Cal(i:(i + m), :));
                
                % Normalize the validation data using the calibration stats
                XVal = normalize(X_Cal((i + m + 1), :), 'Center', mu, 'scale', sig);

                % Center the Y matrices for calibration
                yTrain = y_Cal(i:(i + m), :) - mean(y_Cal(i:(i + m), :));
                
                % Center the validation target using the calibration mean
                yVal = y_Cal((i + m + 1), :) - mean(y_Cal(i:(i + m), :));

                % Compute the PLS model
                [~, ~, ~, ~, B, MSE, stats] = plsregress(XTrain, yTrain, j);
                model(m).ncomp(j).segment(i).B = B;
                model(m).ncomp(j).segment(i).MSE = MSE;
                model(m).ncomp(j).segment(i).stats = stats;

                % Calculate R² for the training data
                yfitPLS_train = [ones(size(XTrain, 1), 1), XTrain] * B;
                TSSRes = sum((yTrain - mean(yTrain)).^2);
                RSSRes = sum((yTrain - yfitPLS_train).^2);
                model(m).ncomp(j).R2(i) = 1 - RSSRes / TSSRes;

                % Calculate Q² for the validation data
                yfitPLS_val = [ones(size(XVal, 1), 1), XVal] * B;
                PRESS = sum((yVal - yfitPLS_val).^2);
                model(m).ncomp(j).Q2(i) = 1 - PRESS / TSSRes;

                % Store the residual for later analysis
                model(m).ncomp(j).residuals(i) = yVal - yfitPLS_val;

                % Store the regression coefficients for each segment
                model(m).ncomp(j).B(i, :) = B';
            end

            % Handle NaN and -Inf values for Q²
            model(m).ncomp(j).Q2(isnan(model(m).ncomp(j).Q2)) = 0;
            model(m).ncomp(j).Q2(model(m).ncomp(j).Q2 == -Inf) = 0;

            % Calculate mean R² and mean Q² across segments
            model(m).ncomp(j).meanR2 = nanmean(model(m).ncomp(j).R2);
            model(m).ncomp(j).meanQ2 = nanmean(model(m).ncomp(j).Q2);
        end
    end
end
