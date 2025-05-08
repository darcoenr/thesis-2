source cluster.sh ../datasets fasta_files/sequences.fasta sequences
rm ../datasets/sequences.fasta

source cluster.sh ../datasets germline.fasta germline
rm ../datasets/germline.fasta

source cluster.sh ../datasets random.fasta random
rm ../datasets/random.fasta

source cluster.sh ../datasets sequences_only_human.fasta sequences_only_human
rm ../datasets/sequences_only_human.fasta

source cluster.sh ../datasets germline_only_human.fasta germline_only_human
rm ../datasets/germline_only_human.fasta

source cluster.sh ../datasets random_only_human.fasta random_only_human
rm ../datasets/random_only_human.fasta