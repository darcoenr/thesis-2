# Germline-aware deep learning models and benchmarks for predicting antibody VH–VL pairing
Repository under construction. Switch to develop to access the code.

# Data availability

Data is available [here](https://zenodo.org/records/17389656?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjM0M2Y3Nzk2LWU3NmItNDA0MC1iMmVlLWM3YjRiZjQ0MmU2OCIsImRhdGEiOnt9LCJyYW5kb20iOiI0ZTYzZDg0NGY0ZDNjMzRkOWQyYTIzYTI1YzA3YmRkOSJ9.7zdkeilG57P8wmVWusUUKURrUusuPYQwIGcrGhfzhsfkzP6I7CbOHuKTBi-jnJZsafOcWrEAz6ZaGlii1tsf2A)

Directory structure:

- **`all_datasets/`** — main folder containing all dataset versions  
  - **`dataset_old/`** — legacy dataset directory  
    - **`classificator/`** — data grouped by classification type  
      - `germline_all/` — contains germline sequences (complete set)  
      - `germline_v/` — variant germline sequences  
      - `random/` — randomized samples for control or comparison  
    - **`clustering/`** — clustering results and related data  
      - `sequences.tsv` — clustered sequence data in tabular format  
    - **`fasta/`** — FASTA files with raw sequence data  
    - **`sequences/`** — processed sequence datasets  
      - `sequences.csv` — main dataset of all sequences  
      - `representative.csv` — representative subset of sequences  
      - `train.csv`, `val.csv`, `test.csv` — training, validation, and testing splits  
    - **`test/`** — evaluation datasets for validation and testing  
      - `val.csv`, `test.csv` — validation and test files used in model evaluation

# Data processing

Download a set of OAS paired data chunk and store them into a repository. To generate the three datasets

# Model usage
