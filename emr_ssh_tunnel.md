# zepplin notebooks
ssh -i ~/ssh/storably-prod.pem -N -L 8157:###.##.##.###:8890 hadoop@###.##.##.###
http://localhost:8157

# ganglia dash
ssh -i ~/ssh/storably-prod.pem -N -L 8158:###.##.##.###:80 hadoop@###.##.##.###
http://localhost:8158/ganglia/