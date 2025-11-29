%% Running the rating analysis and plotting figures
%#ok<*HIST>

% Loading the clean and concatenated data
load("Data\Ratings.mat") 
meana = mean(arousal_all, 'omitnan'); % average arousal rating for each sound
meane = mean(emotionality_all, 'omitnan'); % average emotionality rating for each sound
meanr = mean(recognition_all, 'omitnan'); % average recognition rating for each sound
norma = normalize(meana); % z-scores of arousal
norme = normalize(meane); % z-scores of emotionality
normr = normalize(meanr); % z-scores of recognition


%% Running the reliability analysis
load("Data\Ratings_Long.mat")

ICC2K(TArousal) % ICC for arousal data
ICC2K(TEmotionality) % ICC for emotionality data
ICC2K(TRecognition)% ICC for recognition data


%% Figure 1: Histogram of the average sound rating
% Plotting histogram for each of the average ratings for each sound
bins = 1:0.5:10; % Set the bins for the histogram
figure(1)
hist(meana,bins) 
xlim([1,10])
h = findobj(gca,'Type','patch');
h.FaceColor = [122/255 181/255 29/255]; % bin colours
set(gca,'fontsize',18)
figure(2)
hist(meane,bins)
xlim([1,10])
h = findobj(gca,'Type','patch');
h.FaceColor = [122/255 181/255 29/255]; % bin colours
set(gca,'fontsize',18)
figure(3)
hist(meanr,bins)
xlim([1,10])
h = findobj(gca,'Type','patch');
h.FaceColor = [122/255 181/255 29/255]; % bin colours
set(gca,'fontsize',18)



%% Run the first step of the algorithm
% Loading the concatenated text data
load("Data\Text_Data.mat")
load("Data\File_Order.mat")
% Clean the data and run the first step of the text analysis
[similarityAll, allSillChoice, silhouetteAvg] = textAnalysis(textData);


%% Figure 2: Create a histogram of the optimal number of categories in each
% iteration of the first step of the algorithm
histogram(allSilChoice, 5:1:25, "FaceColor", [122/255 181/255 29/255], "FaceAlpha",1)  % create histogram


%% Figure 3: Performing the k-means on the similarity matrix a few times to
% figure out the optimal number of clusters to use for the second step.
MaxClust = 25; % Maximum number of clusters to test
avgSilhouetteSecondStep = zeros([1,MaxClust]); % Initializing avg sil vector 
% Runing the k-means multiple times
for k = 2:MaxClust
    idx = kmeans(similarityAll, k, 'Replicates', 5);
    silhouetteScores = silhouette(similarityAll, idx);
    avgSilhouetteSecondStep(k) = mean(silhouetteScores);
end

% Figure 3 plotting
plot(avgSilhouetteSecondStep, '-o', 'Color', 'black', 'MarkerFaceColor', 'w'); 

[value,idxbest] = max(avgSilhouetteSecondStep);
xline(idxbest, '--k', 'LineWidth', 1.5); 
% Black circle around it
hold on
plot(idxbest, value, 'ko', 'MarkerSize', 12, 'LineWidth', 2);

% Smaller red circle inside
plot(idxbest, value, 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
set(gca,'fontsize', 12);


%% Figure 4: Creating a t-sne plot for the sounds
Y = tsne(similarityAll);

% Define 10 distinguishable colors
colors = parula(10); 

% Plot with larger dots
gscatter(Y(:,1), Y(:,2), idxAll, colors, '.', 20); % plotting figure 4

legend('Location','northwest'); % keep legend in top-left
set(gca,'fontsize', 18);


%% Figure 5: Creating the dendrogram figure
% Perform hierarchical clustering
Z = linkage(similarityAll, 'ward');
% transforming the similarity matrix into log similarity this is a
% visualization choice mainly to highlight the hierarchical structure of
% the sound clustering
Zlog = Z;
Zlog(:,3) = log(Zlog(:,3)+1.1);
figure('Position',[100 100 2000 800]); % create empty figure
[H,T,outperm] = dendrogram(Zlog, 0, 'Labels', audioNames); % create dendrogram
% Creating the label structure seen in figure 5 
set(gca, 'FontSize', 8);
xtickangle(90);
set(gca,'TickLength',[0 0]); 

% Get current labels
ax = gca;
labels = ax.XTickLabel;


n = numel(labels);
newLabels = labels;  % initialize

for i = 1:n
    layer = rem(i-1, 5)+ 1;
    switch layer
        case 1
            newLabels{i} = [labels{i}, ' $\rule[0.5ex]{0.5cm}{0.2mm}$'];   % top layer
        case 2
            newLabels{i} = [labels{i}, ' $\rule[0.5ex]{1.5cm}{0.2mm}$'];    % middle upper
        case 3
            newLabels{i} = [labels{i}, ' $\rule[0.5ex]{2.5cm}{0.2mm}$'];    % middle lower
        case 4 
            newLabels{i} = [labels{i}, ' $\rule[0.5ex]{3.5cm}{0.2mm}$'];    % bottom
        case 5 
        newLabels{i} = [labels{i}, ' $\rule[0.69ex]{4.5cm}{0.2mm}$'];    % bottom


    end
end

ax.XTickLabel = newLabels;
set(gca,'TickLabelInterpreter','latex')
xtickangle(90);  % keep rotation if desire
ylabel('log(dissimilarity)')


%% Table 1
% Perform the second step of the algorithm
idxAll = kmeans(similarityAll, 10,"Replicates",10);

% Create an empty table
CategoriesAll(1:40,1:10) = "";
% Populate the table with the sound names
for i = 1:10
    whichAudios = find(idxAll == i);
    tempNames = audioNames(whichAudios);
    for ii = 1:length(tempNames)
        CategoriesAll(ii,i) = string(tempNames{ii}); 
    end
end