# -*- mode: ruby -*-
# vi: set ft=ruby :

$script = <<SCRIPT
mkdir -p /home/vagrant/data/mnist
cd /home/vagrant/data/mnist
wget --quiet http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget --quiet http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget --quiet http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget --quiet http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
chown -R vagrant:vagrant /home/vagrant
SCRIPT

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/trusty32"

  config.vm.box_url = "https://vagrantcloud.com/ubuntu/trusty32"

  config.vm.provider "virtualbox" do |v|
    v.memory = 2048
    v.cpus = 2
	v.gui = true
  end

  # config.vm.synced_folder "vm", "/home/vagrant"

  config.vm.provision :puppet do |puppet|
    puppet.module_path = "modules"
  end

  config.vm.provision "shell", inline: $script

end

