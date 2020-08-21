'''
    Dockerfile 
    Author: Lin Meng
'''
# official image in Docker Hub
FROM pytorch/pytorch:latest 

# Set the working directory
WORKDIR /gnn


# Run the command inside your image filesystem.
RUN pip install numpy dill scipy pandas sklearn matplotlib networkx
RUN pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install torch-geometric

# copy all files in /home/code/isonode on Host to docker container filesystem
COPY . .
# Add metadata to the image to describe which port the container is listening on at runtime.
# EXPOSE 8080

# Run the specified command within the container.
# CMD [ "npm", "start" ]

