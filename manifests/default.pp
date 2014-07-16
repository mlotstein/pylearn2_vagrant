exec { "apt-update":
        command     => "/usr/bin/apt-get update"
} ->
package { [
        "build-essential",
        "gfortran",
        "python",
        "python-dev",
        "python-numpy",
        "python-scipy",
        "python-pip",
        "python-nose",
        "python-yaml",
        "python-imaging",
        "python-matplotlib",
        "libopenblas-dev",
        "liblapack-dev",
        "git",
        "mongodb",
        "python-networkx",
        "python-protobuf",
        "openjdk-7-jre-headless",
        "python-psutil",
        "vim"
    ]:
    ensure => latest,
} ->
package { "git+git://github.com/Theano/Theano.git":
  install_options => "--no-deps",
  ensure => installed,
  provider => 'pip',
} ->
vcsrepo { "/home/vagrant/pylearn2":
  ensure   => present,
  provider => git,
  source   => 'https://github.com/lisa-lab/pylearn2.git',
} ->
exec { 'install pylearn2':
  command   => '/usr/bin/python setup.py develop',
  cwd      => '/home/vagrant/pylearn2',
} ->
vcsrepo { "/home/vagrant/HPOlib":
  ensure   => present,
  provider => git,
  source   => 'https://github.com/mblum/HPOlib.git',
} ->
exec { 'install HPOlib':
  command   => '/usr/bin/python setup.py install',
  cwd      => '/home/vagrant/HPOlib',
} ->
file { "/home/vagrant/data":
  ensure => "directory",
} ->
exec { 'add pylearn2 path':
  command   => '/bin/sed -i "\$aexport PYLEARN2_DATA_PATH=/home/vagrant/data" /etc/bash.bashrc',
  unless    => '/bin/grep "export PYLEARN2_DATA_PATH=/home/vagrant/data" -c /etc/bash.bashrc',
}
