import asyncio
import json
import numpy as np
import websockets
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetargeterClient:
    def __init__(self, uri="ws://localhost:8765"):
        self.uri = uri
        self.websocket = None
        
    async def connect(self):
        self.websocket = await websockets.connect(self.uri)
        logger.info(f"Connected to {self.uri}")
        
    async def close(self):
        if self.websocket:
            await self.websocket.close()
            logger.info("Connection closed")
            
    async def initialize_retargeter(self, model_path, urdf_path, source="avp"):
        """Initialize the retargeter with model and URDF paths"""
        if not self.websocket:
            await self.connect()
            
        message = {
            "command": "init",
            "model_path": model_path,
            "urdf_path": urdf_path,
            "source": source
        }
        
        await self.websocket.send(json.dumps(message))
        response = await self.websocket.recv()
        return json.loads(response)
        
    async def retarget(self, mano_data):
        """Send MANO data for retargeting and get joint angles"""
        if not self.websocket:
            await self.connect()
            
        json_data = {}
            
        for key, value in mano_data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value
                
        json_data = json.dumps(json_data)
    
        message = {
            "command": "retarget",
            "mano_data": json_data
        }
        
        start_time = time.time()
        await self.websocket.send(json.dumps(message))
        response = await self.websocket.recv()
        end_time = time.time()
        
        round_trip_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"Round-trip time: {round_trip_time:.2f} ms")
        
        return json.loads(response)

    def convert_to_numpy(self, data):
        """Convert data to numpy array"""
        if isinstance(data, list):
            return np.array(data)
        return data

async def main():
    # Example usage
    client = RetargeterClient()
    
    try:
        # Initialize retargeter
        init_response = await client.initialize_retargeter(
            model_path="/home/ccc/orca_ws/src/orca_configs/orcahand_v1_right_clemens_stanford",
            urdf_path="/home/ccc/orca_ws/src/orca_ros/orcahand_description/models/urdf/orcahand_right.urdf"
        )
        logger.info(f"Initialization response: {init_response}")
        
        if init_response['status'] == 'success':
            # Load AVP data
            file_path = "/home/ccc/orca_ws/src/orca_retargeter/orca_retargeter/example_data/avp_data/back24.json"
            
            with open(file_path, "r") as f:
                data_log = json.load(f)
            
            timestamps = data_log["timestamps"]
            data = data_log["data"]
            
            # Process each frame
            for i, timestamp in enumerate(timestamps):
                logger.info(f"Processing frame {i} at timestamp {timestamp}")
                
                # Convert frame data to numpy arrays, this is a dictionary of numpy arrays, equivalent to what would come from AVP
                frame_data = {k: client.convert_to_numpy(v[i]) for k, v in data.items()}
                
                # Perform retargeting
                retarget_response = await client.retarget(frame_data)
                
                if retarget_response['status'] == 'success':
                    target_angles = retarget_response['target_angles']
                else:
                    logger.error(f"Frame {i} retargeting failed: {retarget_response['message']}")
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 