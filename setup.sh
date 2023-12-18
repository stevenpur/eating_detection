# set up the environment for the project

# check if conda environment already exists
if conda env list | grep -q 'eating_detect'; then
    echo "eating_detect environment already exists"
else
    # create conda environment
    echo "creating eating_detect environment"
    conda create -n eating_detect python=3.11 git pip openjdk
    pip install -r requirements.txt
fi

# activate conda environment
conda activate eating_detect