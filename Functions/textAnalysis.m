function [varargout] = textAnalysis(data,varargin)
% textAnalysis  Perform text cleaning and first-step clustering to compute a similarity matrix
%
%   similarityAll = textAnalysis(data) cleans participant-generated text
%   responses and runs the first step of the two-stage semantic
%   categorization analysis. The function returns a SoundNum-by-SoundNum
%   similarity matrix describing how consistently two sounds are placed in
%   the same semantic cluster across iterations.
%
%   [similarityAll, allSilChoice, silhouetteAvg] = textAnalysis(data, ...)
%   also returns the selected number of clusters for each iteration and the
%   silhouette scores for all tested cluster numbers.
%
% INPUTS
%
%   data        - A matrix of strings of size ParticipantNum-by-SoundNum.
%                 Each entry contains the participant’s textual label or
%                 guess for that sound. Empty or missing responses should be
%                 represented as "".
%
%   numiter     - (Optional) Number of iterations for the first-step
%                 clustering procedure. In each iteration, the function
%                 embeds all unique cleaned guesses into a semantic space
%                 and runs k-means clustering to identify the optimal
%                 cluster solution.  
%                 Default: 1000
%
%   clusterRange - (Optional) Two-element vector specifying the range of
%                  cluster numbers to test during each iteration. The
%                  function computes silhouette scores for each cluster
%                  solution in this range and selects the cluster number
%                  with the highest silhouette value.  
%                  Default: [5 25]
%
%   randseed    - (Optional) Random seed for reproducibility.  
%                 Default: 1
%
%   stopwords   - (Optional) List of stop words to remove from text
%                 responses. These strings will be replaced by empty
%                 entries before embedding.  
%                 Default:  
%                 [stopWords('Language','en'), "voice","noise","sound", ...
%                  "noises","sounds","voices","answer"]
%
% OUTPUTS
%
%   similarityAll - A SoundNum-by-SoundNum matrix. Element (i,j) represents
%                   the normalized number of times sound i and sound j
%                   were assigned to the same k-means cluster across all
%                   iterations.
%
%   allSilChoice  - A numiter-by-1 vector containing the chosen number of
%                   clusters for each iteration (i.e., the cluster count
%                   with maximum silhouette score in that iteration).
%
%   silhouetteAvg - A matrix storing the silhouette score curves for all
%                   tested cluster numbers across all iterations.
%
% DESCRIPTION
%
%   This function implements the first step of the semantic categorization
%   pipeline described in the study. All participant guesses are cleaned
%   (stop-word removal, deletion of nonsense entries, autocorrection)
%   and then embedded into a 200-dimensional GloVe semantic space. In each
%   iteration, unique embedded guesses are clustered using k-means across a
%   specified range of cluster numbers. The optimal cluster solution is
%   selected via silhouette analysis. The full set of guesses is then used
%   to assign each sound to a cluster for that iteration. Repeating this
%   procedure numiter times yields a consensus-based similarity matrix
%   measuring how often each pair of sounds co-clusters.
%
%   This matrix serves as input to the second-stage clustering procedure
%   described in the main analysis pipeline.
%
% See also: kmeans, silhouette, stopWords

numiter = 1000;
clusterRange = [5,25];
randseed = 1;

if nargin >= 2
    if ~isempty(varargin{1})
        numiter = varargin{1};
    end
end

disp(['Chosen number of iterations: ' num2str(numiter)])

if nargin >= 3
    if ~isempty(varargin{2})
        clusterRange = varargin{2};
    end
end

if ~numel(clusterRange) == 2
    error('Range of clusters argument must contain a lower and upper bound')
end

if clusterRange(1) > clusterRange(2)
    error('Cluster range lower bound should be the first entry')
end

if nargin >= 4
    if ~isempty(varargin{3})
        randseed = varargin{3};
    end
end


disp('Loading embedding space')
% Loading the glove twitter word embedding
try
    emb = readWordEmbedding('glove.twitter.27B.200d.txt');
catch ME
    error(['Failed to load the GloVe embedding file. ', ...
           'Please ensure that the file exists and that the path is correct.\n', ...
           'Specified path: %s\nOriginal error: %s'], 'glove.twitter.27B.200d.txt', ME.message);
end



% Define stopwords list (MATLAB's built-in stopwords list)
stopwords = stopWords('Language','en');
% Adding our own list of stop words
% Perhaps add an option for the user to enter their own stop words here?
stopwords = [stopwords "voice" "noise" "sound" "noises" "sounds" "voices" "answer"];

if nargin >= 5
    stopwords = [varargin{5}];
end

% Tokenize the data
documents = tokenizedDocument(data);
% Turn every word into lower case
documents = lower(documents);
% Apply MATLAB's autocorrect function
documents = correctSpelling(documents);
% Remove stop words
documents = removeWords(documents, stopwords);

% Initialize an empty cell to store embeddings
docVectors = cell(numel(documents), 1);
disp('Pre-processing text data')
% Pre-processing of the text data
for i = 1:numel(documents)
    if strcmp(data(i), "No answer") % Answers that were left empty were automatically filled with No answer
        docVectors{i} = single(zeros(1, emb.Dimension)); % Handle missing data
        continue
    end
    words = string(documents(i));
    wordVectors = word2vec(emb, words); % Retrieve word vectors
    wordVectors = wordVectors(~any(isnan(wordVectors), 2), :); % Remove missing words
    if ~isempty(wordVectors)
        % Average the word vectors for each document
        % This is a quick way to handle guesses with multiple words
        docVectors{i} = mean(wordVectors, 1);
    else
        docVectors{i} = single(zeros(1, emb.Dimension)); % Handle documents with no match
    end
end


% Selecting unique and non-empty guesses words only
docMatrix = cell2mat(docVectors); % Convert to matrix format
% Selecting the non-empty guesses first:
% Turning matrix to logical with 1 when any value in the matrix is not empty
nonEmptyData = docMatrix ~= 0; 
% Turning matrix into vector of 1 iff all values in that row is empty (i.e. No Answer)
emptyVec = zeros([1,length(nonEmptyData)]);
for i = 1:length(nonEmptyData)
    emptyVec(i) = all(nonEmptyData(i,:) == 0);
end
% Only 1 iff all values in that row is not empty
nonEmptyDataRow = ~emptyVec;
% Selecting non-empty guess
fullMat = docMatrix(nonEmptyDataRow,:);
% Saving the index of the empty guesses
emptyIdx = find(emptyVec);
% Creating a function that will insert these empty guesses back
% This will be used after running the k-means
insert = @(n,x,a)  [x(1:a-1), n, x(a:end)];
% Creating the matrix with the unique guesses only
[X, ~, ic] = unique(fullMat , 'rows');

% Performing the first step of the algorithm
disp('Running the first step of the k-means')
sumSim = zeros([length(data) length(data)]);
silhouetteAvg = zeros([numiter,clusterRange(2)]);
for itr = 1:numiter
    % Perform K-means clustering
    disp(['Iteration Number: ' num2str(itr)])
    % Choosing the number of clusters for this iteration
    avg = zeros([1,clusterRange(2)]);
    % Running the algorithm multiple times to check which number of
    % clusters results in the maximum number of clusters
    for r = clusterRange(1):clusterRange(2)
        rng(itr  * randseed) % Set random seed
        idx = kmeans(X, r, 'Replicates', 5,'Distance','cosine');
        silhouetteScores = (silhouette(gather(X), gather(idx)));
        avg(r) = mean(silhouetteScores);
    end
    % Save the average silhouette score for every run of the k-means
    silhouetteAvg(itr,:) = avg;
    % Find the run that produced the highest avg silhouette score
    numClusters = find(avg == max(avg(clusterRange(1):clusterRange(2))), 1);
    disp(['Chosen Number of Clusters: ' num2str(numClusters)])
    % Run the k-means again with the chosen number of clusters
    rng(itr * randseed)
    idxMain = kmeans(X, numClusters, 'Replicates', 5, 'Distance','cosine');
    % Insert the non-unique guesses
    OGIDX = zeros([1,length(ic)]);
    for ii = 1:length(ic)
        OGIDX(ii) = idxMain(ic(ii));

    end
    idxMain = OGIDX;
    % Insert the 'No Answer' guesses as zeros
    for ii = 1:length(find(emptyVec))
        idxMain = insert(0,idxMain,emptyIdx(ii));
    end
    % Reshape the classification of the guesses into the original format
    % (Participant x Guess)
    idxMain = idxMain';
    idxReshaped = reshape(idxMain,size(data,1),size(data,2));
    % Count the number of times a sound had a guess in a certain cluster
    sumsOfCluster = zeros([numClusters, size(data,2)]);
    for i = 1:numClusters
        for ii = 1:size(data,2)
            sumsOfCluster(i,ii) = gather(sum(idxReshaped(:,ii) == i));
        end
    end
    % Find the cluster that the guesses of each of the sounds was
    % classified to the most and take that as the classification for that
    % sounds
    [~, idxSum] = max(sumsOfCluster, [], 1);
    % Create a sound x sound matrix which contains information about which
    % sound was classified with another one. For example: if the value at
    % row 2 and column 3 is 1, then sound 2 and sound 3 were classified in
    % the same category
    similarity = zeros(size(data,2));
    for i = 1:length(idxSum)
        similarity(i,:) = idxSum == idxSum(i);
    end
    % Save that matrix for each iteration
    sumSim = sumSim + similarity;

end

allSilChoice = zeros([1,numiter]);

% Save the chosen number of clusters for each iteration
for i = 1:length(silhouetteAvg)
    allSilChoice(i) = find(silhouetteAvg(i,:) == max(silhouetteAvg(i,clusterRange(1):clusterRange(2))), 1);
end

% Create a similarity matrix by dividing the resulting matrix from last
% step with the number of iterations. This is easier to use with
% classification algorithms
similarityAll = sumSim/numiter;

varargout{1} = similarityAll;
varargout{2} = allSilChoice;
varargout{3} = silhouetteAvg;

end