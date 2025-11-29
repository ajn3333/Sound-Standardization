function ICC_2_k = ICC2K(T)
% T should be a table of the long format of the ratings with variables:
%   T.rating   : numeric rating
%   T.soundID  : categorical or numeric id for each sound (1..291)
%   T.raterID  : categorical or numeric id for each participant
%
% Example:
% T = table(ratingsVec, soundVec, raterVec, 'VariableNames', {'rating','soundID','raterID'});

% ---- Fit LME ----
lme = fitlme(T, 'rating ~ 1 + (1|soundID) + (1|raterID)', 'FitMethod', 'REML');


varSound = lme.covarianceParameters{1}; 
varRater = lme.covarianceParameters{2};  
varResidual = (std(lme.residuals, 'omitmissing'))^2;


nSounds = length(unique(table2array(T(:,3))));

% Preallocate
ratingsPerSound = zeros(nSounds,1);

for i = 1:nSounds
    tempIdx = table2array(T(:,3)) == i;
    ratingsPerSound(i) =  sum(~isnan(table2array(T(tempIdx,1))));
end

% Mean number of raters per sound
k = mean(ratingsPerSound);

ICC_2_k = varSound / (varSound + (varRater + varResidual)/k);


end