### This project structure:
- four algorithms folder: DSSM, LightGCN, SASRec, and GRU4Rec.
- In the tool folder, code including related process, split rule, dataset, dataloader, evaluator and filter rule.
- In the code folder, code transforming modal information into an embedding 
- In the DataSet, store the dataset we use.
### for any  algorithms folder, include:
- baselilne algorithms.
- strength with modal information(title, cover, title+cover).
- a final version incorporating all the different types above (can be switched with a specific command and flag)

### all functionalities have now been integrated into the following four files:
- DSSM_WITH_MODAL, GRU4Rec_WITH_MODAL, SASRec_WITH_MODAL, and LightGCN_WITH_MODAL.
### The fusion strategy and fused content can be controlled via the following two configuration fields:
- FUSION_MODE can be set to 'base' | 'early' | 'late1' | 'late2', which determines the fusion strategy used by the model.
- CURRENT_MODAL can be set to COVER, TITLE, or COVER-TITLE to switch between different types of multimodal content used for fusion.

- FUSION_MODE controls the fusion strategy:
  - 'base': unimodal baseline without multimodal fusion
  - 'early', 'late1', 'late2': different fusion strategies discussed in the dissertation
- CURRENT_MODAL controls the multimodal content used for fusion.
### Running the experiments 
All experiments are executed by running the corresponding *_WITH_MODAL file for each model.
Fusion strategies and multimodal inputs can be controlled via configuration flags.

### Results
All training records and experimental results are saved in the file **"Train Record"**.
Results labeled as **validation mode** are obtained on the validation set and are used for hyperparameter selection.  
Results labeled as **train mode** are evaluated on the test set and represent the final model performance.
