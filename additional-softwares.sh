set -e

# Change the packages to your requirement

# Add the packages that are difficult to install from the conda environment

# Do not remove flask

echo "Installing Flask..."
pip install flask

echo "Installing Gunicorn..."
pip install gunicorn

echo "Install torch..."
pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html --quiet
pip install fair-esm
pip install torch-geometric
conda install -c bioconda viennarna
