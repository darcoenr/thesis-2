DIR=$1
FILE=$2
OUT_NAME=$3
MIN_SEQ_ID=$4

# Create database
mkdir $DIR'/DB'
mmseqs createdb $DIR'/'$FILE $DIR'/DB/DB'

# Cluster
mkdir $DIR'/cluster'
# This command uses Cascade Clustering approach
mmseqs linclust $DIR/DB/DB $DIR'/cluster/cluster' $DIR/cluster/tmp --min-seq-id $MIN_SEQ_ID
mmseqs createtsv $DIR'/DB/DB' $DIR'/DB/DB' $DIR'/cluster/cluster' $DIR/$OUT_NAME'.tsv'

# Clean
rm -rf $DIR/DB
rm -rf $DIR/cluster