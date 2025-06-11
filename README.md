# Orca Hand Retargeter

A Python package for retargeting MANO hand poses to Orca Hand joint angles. This package provides both a direct Python interface and a WebSocket server for isolated usage.

## Installation

1. Create a new virtual environment:
```bash
python -m venv retargeter_env
source retargeter_env/bin/activate  # On Linux/Mac
# or
.\retargeter_env\Scripts\activate  # On Windows
```

2. Install the required packages:
```bash
pip install numpy==1.24.3
pip install torch==2.1.0
pip install pytorch-kinematics==0.0.1
pip install scipy==1.10.1
pip install pyyaml==6.0.1
```

## Basic Usage

```python
from orca_retargeter import Retargeter

# Initialize retargeter
retargeter = Retargeter(
    hand="right",  # or "left"
    urdf_path="/path/to/hand/urdf",
    source="avp"  # or "rokoko"
)

# Retarget MANO data
mano_data = {
    "right_wrist": wrist_matrix,  # (4, 4) transformation matrix
    "right_fingers": fingers_matrix,  # (21, 4, 4) transformation matrices
}

target_angles, debug_dict = retargeter.retarget(mano_data)
```

## WebSocket Interface

For isolated usage or when you need to run the retargeter in a separate environment with strict dependency requirements, you can use the WebSocket interface.

### Additional WebSocket Dependencies
```bash
pip install websockets==12.0
```

### Starting the WebSocket Server
```bash
python -m orca_retargeter.websocket.server
```

### Using the WebSocket Client
```python
import asyncio
from orca_retargeter.websocket.client import RetargeterClient

async def main():
    client = RetargeterClient()
    
    try:
        # Initialize retargeter
        init_response = await client.initialize_retargeter(
            model_path="/path/to/model/config",
            urdf_path="/path/to/hand/urdf"
        )
        
        if init_response['status'] == 'success':
            # Send MANO data for retargeting
            retarget_response = await client.retarget(mano_data)
            
            if retarget_response['status'] == 'success':
                target_angles = retarget_response['target_angles']
                print(f"Retargeted angles: {target_angles}")
                
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```


## License

MIT License
