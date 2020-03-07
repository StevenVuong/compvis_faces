#!/bin/bash
set -euo pipefail

# SSH into gcloud instance
gcloud beta compute --project "hubble-tools" ssh --zone "europe-west4-a" "opensfm-2"

# install docker
# https://stackoverflow.com/questions/41133455/docker-repository-does-not-have-a-release-file-on-running-apt-get-update-on-ubun
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get install docker-ce docker-ce-cli containerd.io

# install docker compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# make ssh creds
ssh-keygen -t rsa -b 4096 -C "steven@hackpartners.com"
cat ~/.ssh/id_rsa.pub
# add to https://github.com/settings/ssh/new

# clone project and go to right branch
git clone --recurse-submodules https://github.com/HackPartners/hubble-kkom.git
cd hubble-kkom/
git checkout OpenSfM


# mount (full storage)
# https://cloud.google.com/storage/docs/gcs-fuse
# https://groups.google.com/forum/#!topic/gce-discussion/jxl11PIsbhM
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update
sudo apt-get install gcsfuse
sudo usermod -a -G fuse $USER
exit

# ssh back in
gcloud beta compute --project "hubble-tools" ssh --zone "europe-west4-a" "opensfm-2"

# fuseeee
mkdir gcs-bucket
gcsfuse public.hubble.tools ~/gcs-bucket/

# copy data to required dir
cp -r ~/fuse-bucket/example-opensfm-inputs/demo_vid.mp4 ~/.
#cd ~/hubble-kkom/cli-apps/playground/sandbox/local-storage/input/
#gsutil cp gs://public.hubble.tools/example-opensfm-inpits/ /.

# start container
cd ~/hubble-kkom/cli-apps/playground
touch requirements.txt
./docker-start.bash

# run opensfm command
cd ~/src/open-sfm/bash
./run_all.bash -d 145 -f 10 -v /root/local-storage/input/example-opensfm-inpits/videos_sbsd9p8134.mp4
./run_all.bash -s 173 -d 480 -f 10 -v /root/local-storage/input/example-opensfm-inpits/videos_k4cnze0rk4.mp4
hello


# copy over a large number of files
# gsutil -m cp -n -r ~/hubble-kkom/cli-apps/playground/sandbox/local-storage/output/video-1-2-cut145/ gs://public.hubble.tools/example-opensfm-outputs/
cp -n -r /root/local-storage/output/example-opensfm-inpits/videos_sbsd9p8134.mp4 ~/gcs-bucket/example-opensfm-outputs/

# delete unnecesary files in depthmaps dir
# https://linuxconfig.org/bash-bin-rm-argument-list-too-long-solution
for i in *.pruned.npz.ply; do sudo rm -rf "$i"; done 
