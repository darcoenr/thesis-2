# Germline-aware deep learning models and benchmarks for predicting antibody VH–VL pairing

# Data availability

Data is available [here](https://zenodo.org/records/17389656?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjM0M2Y3Nzk2LWU3NmItNDA0MC1iMmVlLWM3YjRiZjQ0MmU2OCIsImRhdGEiOnt9LCJyYW5kb20iOiI0ZTYzZDg0NGY0ZDNjMzRkOWQyYTIzYTI1YzA3YmRkOSJ9.7zdkeilG57P8wmVWusUUKURrUusuPYQwIGcrGhfzhsfkzP6I7CbOHuKTBi-jnJZsafOcWrEAz6ZaGlii1tsf2A)

Directory structure:

- **`all_datasets/`**
  - **`dataset_old/`** 
    - **`classificator/`** — data used to train the classifiers 
      - **`germline_all/`**
      - **`germline_v/`**
      - **`random/`**
    - **`clustering/`** — result of the clustering operation 
      - `sequences.tsv` — clustered sequence data in tabular format  
    - **`fasta/`** — directory used to store FASTA sequences temporarily
    - **`sequences/`** — processed sequence datasets  
      - `sequences.csv` — main dataset of all sequences  
      - `representative.csv` — representative subset of sequences  
      - `train.csv`, `val.csv`, `test.csv` — training, validation, and testing germline-aware splits
    - **`test/`** — evaluation datasets for validation and testing  
      - `val.csv`, `test.csv` — validation and test files used for the test procedure

Each of **`germline_all`**, **`germline_v`** and **`random`** contains the directories **`train`**, **`val`** and **`test`** used to train, validate and test each of the classifiers. For the **`germline_all`** and **`germline_v`** directories, additional files with germline information are included.

# Data processing

Download a set of OAS paired data chunk and store them into a repository. To generate the three datasets

# Model usage

Consult the example.ipynb notebook.
