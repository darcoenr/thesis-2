source cluster.sh ../datasets fasta_files/sequences.fasta ../datasets/clustering-only_human-germline_v/sequences 0.8
source cluster.sh ../datasets fasta_files/random_pairing.fasta ../datasets/clustering-only_human-germline_v/random_pairing 0.8
source cluster.sh ../datasets fasta_files/germline_pairing.fasta ../datasets/clustering-only_human-germline_v/germline_pairing 0.8
rm -rf fasta_files