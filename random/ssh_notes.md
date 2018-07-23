# jupyter notebook running in docker on remote machine
sudo nvidia-docker run -it -p 8889:8888 --ipc=host -v ~/PyReX/notebooks:/home jesse/mxnet:1.2.0
jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=
ssh -N -L 8889:184.105.6.101:8889 paperspace@184.105.6.101


# SPARK
## zepplin notebooks
ssh -i ~/ssh/storably-prod.pem -N -L 8157:###.##.##.###:8890 hadoop@###.##.##.###
http://localhost:8157

## ganglia dash
ssh -i ~/ssh/storably-prod.pem -N -L 8158:###.##.##.###:80 hadoop@###.##.##.###
http://localhost:8158/ganglia/