# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/trusty32"

  config.vm.box_url = "https://vagrantcloud.com/ubuntu/trusty32"

  # config.vm.synced_folder "vm", "/home/vagrant"

  config.vm.provision :puppet do |puppet|
    puppet.module_path = "modules"
  end

end
