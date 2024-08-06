# Flow2Quake

- [Flow2Quake](#flow2quake)
  - [Development](#development)
    - [Environment Setup](#environment-setup)
    - [`dm`: Data Manager (On-site archive access)](#dm-data-manager-on-site-archive-access)
      - [Global Alias](#global-alias)
      - [Using `dm`](#using-dm)
      - [Configuring SSH for `dm`](#configuring-ssh-for-dm)

## Development

### Environment Setup

Running / developing Flow2Quake requires:

- [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://docs.anaconda.com/miniconda/)
- Python 3.10
- [`rsync`](https://rsync.samba.org/)

Create Conda environment from file (currently Linux-only):

```
conda create -f environment-ubuntu.yml -n flow2quake
```

### `dm`: Data Manager (On-site archive access)

```
$ dm -h
usage: dm [-h] {upload-data,download-data,upload-output,download-output,identify,status} ...

Data Manager utility for Flow2Quake (/home/michael/flow2quake)

options:
  -h, --help            show this help message and exit

subcommands:
  {upload-data,download-data,upload-output,download-output,identify,status}
    upload-data         Upload input data for an individual case
    download-data       Download input data for an individual case
    upload-output       Upload most recent outputs (plots, etc) for an individual case
    download-output     Download someone's outputs (plots, etc) for an individual case
    identify            Set your name for upload entries
    status              Check remote store status
```

`dm` is a simple utility that's bundled with the Flow2Quake codebase to sync large data files with GPS' archive and keep track of runs. It's thin wrapper around `rsync` and `scp` that copies data to and from `tecto` while managing paths and versioning for you. 

#### Global Alias

To easily alias `./dm` to `dm` (making it runnable from anywhere), you can run the following from a shell in the repo root folder.

For `bash` users:

```
echo "alias dm=\"_dm(){ local WD=\\\$(pwd);cd $(pwd) && ./dm \\\$@;cd \\\$WD;};_dm\"" >> ~/.bashrc
``` 

For `zsh` users:

```
echo "alias dm=\"_dm(){ local WD=\\\$(pwd);cd $(pwd) && ./dm \\\$@;cd \\\$WD;};_dm\"" >> ~/.zshrc
```

#### Using `dm`

This quickstart uses the Groningen site example (registered by name as `groningen`, located in `flow2quake/cases/groningen`).

First, you should set your name:

```
dm identify
```

For example, you can download the archive data for case `groningen` using:

```
dm download-data groningen
```

If you make any important additions or changes to the input data, you can upload them using:

```
dm upload-data groningen
```

After running the models in `flow2quake/cases/groningen`, you can then archive your results as a unique, timestamped run-output using:

```
dm upload-output groningen
```

The above three commands can have their confirm dialog overwritten using the `-y` flag:

```
# will not ask for confirmation
dm upload-output groningen -y
```

#### Configuring SSH for `dm`

The simplest possible SSH config to use `dm` would look like (paste the following into `~/.ssh/config`):

```
Host *.gps.caltech.edu
    Hostname %h
    User [YOUR USERNAME]
```

In order to access the data archive while off-site without a VPN, your SSH config should allow directly logging in to `tecto.gps.caltech.edu`. If you haven't yet, it's highly recommended that you set up a local SSH key to instantly access `tecto` without manually logging in twice.

An SSH config (`~/.ssh/config`) proxying `tecto` traffic through `earth` might look like the following:

```
Host earth.gps.caltech.edu
    Hostname earth.gps.caltech.edu
    User [YOUR USERNAME]
    ProxyJump none
    Port 22
    IdentityFile ~/.ssh/[YOUR SSH KEY NAME]
    UpdateHostKeys no
    ForwardAgent yes
  
Host *.gps.caltech.edu
    Hostname %h
    User [YOUR USERNAME]
    ProxyJump earth.gps.caltech.edu
    Port 22
    IdentityFile ~/.ssh/[YOUR SSH KEY NAME]
    UpdateHostKeys no
```
