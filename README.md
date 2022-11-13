# LegoSorterServer

Lego Sorter Server provides methods for detecting and classifying Lego bricks.

## How to run
1. Download the repository
   ```sh
   git clone https://github.com/LegoSorter/LegoSorterServer.git
   ```

2. Download network models for detecting lego bricks
   ```sh
   wget https://github.com/LegoSorter/LegoSorterServer/releases/download/1.2.0/detection_models.zip
   wget https://github.com/LegoSorter/LegoSorterServer/releases/download/1.2.0/classification_models.zip
   ```

3. Extract models
   ```sh
   unzip detection_models.zip -d ./LegoSorterServer/lego_sorter_server/analysis/detection/models
   unzip classification_models.zip -d ./LegoSorterServer/lego_sorter_server/analysis/classification/models
   ```

4. Run the start script
   ```sh
   cd ./LegoSorterServer
   bash ./start.sh
   ```

The server is now ready to handle requests. By default, the server listens on port *50051*

## Lego Sorter App
To test **Lego Sorter Server**, use the [Lego Sorter App](https://github.com/LegoSorter/LegoSorterApp), which is an application dedicated for this project.

## How to send a request (Optional)
**Lego Sorter Server** uses [gRPC](https://grpc.io/) to handle requests, the list of available methods is defined in `LegoSorterServer/lego_sorter_server/proto/LegoBrick.proto`.\
To call a method with your own client, look at [gRPC documentation](https://grpc.io/docs/languages/python/basics/#calling-service-methods)
