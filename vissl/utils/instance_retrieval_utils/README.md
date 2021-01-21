### Installing Oxford and Paris datasets

0. Follow instructions at https://github.com/figitaki/deep-retrieval#datasets

```bash
cd ~/local  # change to the desired location
mkdir datasets
cd datasets

# Install the Oxford dataset
mkdir -p Oxford
cd Oxford
mkdir jpg lab
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz
tar -xzf oxbuild_images.tgz -C jpg
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz
tar -xzf gt_files_170407.tgz -C lab
cd ..

# Install the Paris dataset
mkdir -p Paris
cd Paris
mkdir jpg lab tmp
# Images are in a different folder structure, need to move them around
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz
tar -xzf paris_1.tgz -C tmp
tar -xzf paris_2.tgz -C tmp
find tmp -type f -exec mv {} jpg/ \;
rm -rf tmp
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_120310.tgz
tar -xzf paris_120310.tgz -C lab
cd ..
cd ..

# Install the evaluation binary
mkdir evaluation
cd evaluation
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp
sed -i '6i#include <cstdlib>' compute_ap.cpp # Add cstdlib, as some compilers will produce an error otherwise
g++ -o compute_ap compute_ap.cpp
cd ..
```

### Installing Revisited Oxford and Paris datasets (roxford5k, rparis6k)

As SOTA is evaluated on revisted versions of these datasets (see http://cmp.felk.cvut.cz/revisitop/), follow these instructions to download the revisited versions.


```bash
cd ~/local  # change to the desired location
mkdir datasets
cd datasets

# Install the Oxford dataset
mkdir roxford5k
cd roxford5k
mkdir jpg
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz
tar -xzf oxbuild_images.tgz -C jpg
wget http://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/gnd_roxford5k.pkl
cd ..

# Install the Paris dataset
mkdir rparis6k
cd rparis6k
mkdir jpg tmp
# Images are in a different folder structure, need to move them around
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz
tar -xzf paris_1.tgz -C tmp
tar -xzf paris_2.tgz -C tmp
find tmp -type f -exec mv {} jpg/ \;
rm -rf tmp
wget http://cmp.felk.cvut.cz/revisitop/data/datasets/rparis6k/gnd_rparis6k.pkl
cd ..
cd ..
```
