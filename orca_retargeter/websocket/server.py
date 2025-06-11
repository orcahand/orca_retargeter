import asyncio
import json
import numpy as np
from websockets.server import serve
from ..retargeter import Retargeter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetargeterServer:
    def __init__(self):
        self.retargeter = None
        
    async def handle_message(self, websocket):
        async for message in websocket:
            try:
                data = json.loads(message)
                command = data.get('command')
                
                if command == 'init':
                    # Initialize retargeter
                    model_path = data.get('model_path')
                    urdf_path = data.get('urdf_path')
                    source = data.get('source', 'avp')
                    
                    try:
                        self.retargeter = Retargeter(
                            hand=model_path,
                            urdf_path=urdf_path,
                            source=source
                        )
                        await websocket.send(json.dumps({
                            'status': 'success',
                            'message': 'Retargeter initialized successfully'
                        }))
                    except Exception as e:
                        await websocket.send(json.dumps({
                            'status': 'error',
                            'message': f'Failed to initialize retargeter: {str(e)}'
                        }))
                
                elif command == 'retarget':
                    if not self.retargeter:
                        await websocket.send(json.dumps({
                            'status': 'error',
                            'message': 'Retargeter not initialized'
                        }))
                        continue
                    
                    # Get MANO joint positions
                    json_data = json.loads(data.get('mano_data'))
                    
                    mano_data = {}
                    for key, value in json_data.items():
                        if isinstance(value, list):
                            mano_data[key] = np.array(value)
                        else:
                            mano_data[key] = value

                   
                    try:
                        # Perform retargeting
                        target_angles, debug_dict = self.retargeter.retarget(mano_data)
                        
                        # Convert numpy arrays to lists for JSON serialization
                        target_angles = {k: float(v) for k, v in target_angles.items()}
                        
                        json_target_angles = {}
                        for key, value in target_angles.items():
                            json_target_angles[key] = float(value)
                            
                        json_target_angles = json.dumps(json_target_angles)
                      
                        await websocket.send(json.dumps({
                            'status': 'success',
                            'target_angles': target_angles,
                        }))
                    except Exception as e:
                        await websocket.send(json.dumps({
                            'status': 'error',
                            'message': f'Retargeting failed: {str(e)}'
                        }))
                
                else:
                    await websocket.send(json.dumps({
                        'status': 'error',
                        'message': f'Unknown command: {command}'
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    'status': 'error',
                    'message': 'Invalid JSON message'
                }))
            except Exception as e:
                await websocket.send(json.dumps({
                    'status': 'error',
                    'message': f'Server error: {str(e)}'
                }))

async def main():
    server = RetargeterServer()
    async with serve(server.handle_message, "localhost", 8765):
        logger.info("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main()) 