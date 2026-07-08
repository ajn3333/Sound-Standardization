# Sound Classification Analysis Pipeline

This repository contains the full MATLAB pipeline used to analyze participants’ textual labels and ratings of environmental sounds.  
The workflow consists of:

1. **Text-based similarity analysis** (Step 1 of the algorithm)  
2. **Clustering of sounds based on the similarity matrix** (Step 2)  
3. **Reliability analysis using ICC**  
4. **Generation of all figures and tables included in the paper**

---

## Repository Contents

### 1. [textAnalysis.m](Functions/textAnalysis.m) 
A MATLAB function that performs **text cleaning**, **feature extraction**, and **similarity-based clustering** using participant-generated labels.  
It runs Step 1 of the algorithm by iteratively clustering text embeddings and selecting the best solution per iteration based on silhouette scores.

**Outputs:**
- `similarityAll` — Sound × Sound similarity matrix  
- `allSilChoice` — Selected number of clusters per iteration  
- `silhouetteAvg` — Average silhouette scores across tested cluster numbers  

The function supports:
- optional stop-word removal  
- configurable number of iterations  
- adjustable cluster search range  

---

### 2. [ICC2K.m (Function)](Functions/ICC2k.m) 
MATLAB script performing **inter-rater reliability analysis** on participants’ rating data.

It fits a linear mixed-effects model using sound and participant as random effects, allowing estimation of variance components and calculation of:

- **ICC(2,k)** — Reliability of the mean rating per sound  

This script corresponds to the reliability analysis reported in the paper.

---

### 3. [main.m](main.m)
The master script that:

1. Loads and prepares all cleaned data  
2. Runs `textAnalysis` (Step 1)  
3. Performs the clustering analysis (Step 2)  
4. Computes ICC reliability  
5. Generates **all five figures** and **Table 1** from the paper:

- **Figure 1:** Histogram of average ratings  
- **Figure 2:** Histogram of optimal category numbers selected across iterations  
- **Figure 3:** Evaluation of cluster numbers by repeatedly clustering the similarity matrix  
- **Figure 4:** t-SNE embedding of sounds  
- **Figure 5:** Hierarchical dendrogram of the final clustering solution  
- **Table 1 :** Final classification of each sound into its category  

This file is the main entry point for reproducing the full analysis pipeline.

---

## [Data Files](Data/)

Four `.mat` files are included in the Data directory:

- `Text_Data.mat`    — Cleaned matrix of participant-generated textual labels  
- `Rating.mat`       — Cleaned matrix of rating values  
- `File_Order.mat`   — Index file specifying the canonical sound order that the above data followed 
- `Ratings_Long.mat` — Cleaned matrix of rating values in long format. This is used by the ICC2K function
These are required to run `main.m`.

---

## [Results Files](Results/)

Two `.mat` files are included in the Results directory:

- `Ratings_Results.mat`             — Average ratings of the sounds and the normed averages
- `Text_Analysis_Results.mat`       — Categorization of the sounds into the 10 categories

---

## Requirements

- **MATLAB R2020b or later**  
- Toolboxes:
  - Statistics and Machine Learning Toolbox  
  - Text Analytics Toolbox  
- Pretrained **GloVe embeddings**

---

## How to Run

1. Clone the repository:
   ```
   git clone https://github.com/ajn3333/Sound-Standardization.git
   cd Sound-Standardization
   ```

2. Add the repository to your MATLAB path:
   ```matlab
   addpath(genpath('.'));
   ```

3. Run the analysis:

   Run each section of the main.m file 

---

## Citation

If you use this code or adapt it for your own research, please cite our paper `https://doi.org/10.64898/2026.04.16.718910`.

---

## Contact

For questions or collaboration:  
**alnaji@uni-muenster.de**
