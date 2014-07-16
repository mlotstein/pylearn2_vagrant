# Getting started with pylearn2 and HPOlib using Vagrant

- Download and install Virtual Box ([https://www.virtualbox.org/wiki/Downloads](https://www.virtualbox.org/wiki/Downloads)).
- Download and install Vagrant ([http://www.vagrantup.com/downloads.html](http://www.vagrantup.com/downloads.html)).
- Clone the modified pylearn2 in a box from github and run the VM. 
    
        git clone https://github.com/mblum/pylearn2_vagrant.git
        cd pylearn2_vagrant
        vagrant up

- Vagrant will download an Ubuntu virtual machine and install pylearn2 and HPOlib on it. It also downloads the MNIST data into the folder `/home/vagrant/data/mnist`.

- Log into the virtual machine by typing

        vagrant ssh

- You should have a working Ubuntu environment with pylearn2 and HPOlib preinstalled.
- Test your HPOlib installation by running the branin benchmark:

        cd /home/vagrant/HPOlib/benchmarks/branin
        HPOlib-run -o ../../optimizers/smac/smac -s 23
        
- Test your pylearn2 installation by training a convolutional neural network on MNIST:

        cd /vagrant/convnetMNIST
        python convnetMNIST.py

If you already installed the original version of *pylearn2 in a box*, you can merge the latest changes and run `vagrant reload --provision` to update your VM.


