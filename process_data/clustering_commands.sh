source cluster.sh ../datasets/only_human-germline_all fasta_files/sequences.fasta clustering/sequences 0.8
source cluster.sh ../datasets/only_human-germline_all fasta_files/random_pairing.fasta clustering/random_pairing 0.8
source cluster.sh ../datasets/only_human-germline_all fasta_files/germline_pairing.fasta clustering/germline_pairing 0.8
rm -rf ../datasets/fasta_files