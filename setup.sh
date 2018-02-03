# Create directories for datasets
mkdir data data/mnist

# Fetch MNIST dataset
wget http://deeplearning.net/data/mnist/mnist.pkl.gz
gunzip -c *.gz > data/mnist/mnist.pkl

# Fetch 20newsgroups dataset
wget https://github.com/CW-Huang/IFT6135H18_assignment/raw/master/20news-bydate.zip
unzip *.zip -d data/
mv data/20news-bydate data/newsgroups

# Clean
rm *.gz *.zip
