# Standard library imports
import os
import sys
import time
from typing import Dict, Any, Optional, Tuple
import multiprocessing as mp
import configparser
from collections import deque

# Third-party imports
import numpy as np
import torch

# RLGym imports
from rlgym.api import AgentID, StateType
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM

class RLBotAdapter:
    """
    Adapter that allows RLGym to use bots from the RLBot framework.
    """
    def __init__(self, bot_folder_path: str, team: int = 1):
        """
        Initialize an adapter for an RLBot Pack bot.
        
        Args:
            bot_folder_path: Path to the bot's folder in RLBotPack
            team: Team number (0=blue, 1=orange)
        """
        self.bot_folder_path = bot_folder_path
        self.team = team
        self.bot_process = None
        self.conn = None
        self.is_running = False
        self.bot_id = os.path.basename(bot_folder_path)
    
    def start(self):
        """Start the bot process and establish communication"""
        # Find the bot's cfg file
        cfg_file = self._find_bot_cfg_file()
        if not cfg_file:
            raise ValueError(f"Could not find cfg file in {self.bot_folder_path}")
        
        # Parse the cfg file to determine bot type and entry point
        bot_config = self._parse_cfg_file(cfg_file)
        
        # Start the bot process using the appropriate method
        if bot_config['language'].lower() == 'python':
            self._start_python_bot(bot_config)
        else:
            raise ValueError(f"Currently only Python bots are supported, got {bot_config['language']}")
        
        self.is_running = True
    
    def get_action(self, game_state: GameState) -> np.ndarray:
        """
        Get an action from the bot based on the game state.
        
        Args:
            game_state: RLGym GameState object
            
        Returns:
            action: An array of controller inputs
        """
        if not self.is_running:
            raise RuntimeError("Bot is not running. Call start() first.")
        
        try:
            # Convert RLGym GameState to RLBot packet format
            packet = self._convert_gamestate_to_packet(game_state)
            
            # Send packet to bot process
            self.conn.send(packet)
            
            # Receive response (controller inputs) with timeout
            if self.conn.poll(timeout=0.1):  # 100ms timeout
                controller_inputs = self.conn.recv()
                
                # Convert controller inputs to RLGym action format
                action = self._convert_controller_to_action(controller_inputs)
            else:
                # If timeout, return no-op action
                action = np.zeros(8)
                
            return action
            
        except Exception as e:
            print(f"Error getting action from bot: {e}")
            return np.zeros(8)  # Return no-op action on error
    
    def stop(self):
        """Stop the bot process and clean up resources"""
        if self.is_running:
            # Send termination signal
            if self.conn:
                try:
                    self.conn.send({"terminate": True})
                    self.conn.close()
                except:
                    pass
            
            # Terminate process
            if self.bot_process:
                try:
                    self.bot_process.terminate()
                    self.bot_process.join(timeout=5)
                except:
                    # Force kill if needed
                    try:
                        self.bot_process.kill()
                    except:
                        pass
            
            self.is_running = False
    
    def _find_bot_cfg_file(self) -> Optional[str]:
        """Find the bot's cfg file in the bot folder"""
        for file in os.listdir(self.bot_folder_path):
            if file.endswith('.cfg'):
                return os.path.join(self.bot_folder_path, file)
        return None
    
    def _parse_cfg_file(self, cfg_file: str) -> Dict[str, Any]:
        """Parse the bot cfg file to extract important information"""
        config = configparser.ConfigParser()
        config.read(cfg_file)
        
        bot_config = {}
        if 'Locations' in config:
            bot_config['python_file'] = config['Locations'].get('python_file', '')
            bot_config['requirements_file'] = config['Locations'].get('requirements', '')
            
        if 'Details' in config:
            bot_config['name'] = config['Details'].get('name', 'Unknown')
            bot_config['language'] = config['Details'].get('language', 'python')
            bot_config['skill_level'] = config['Details'].get('skill_level', '0.5')
            
        return bot_config
    
    def _start_python_bot(self, bot_config: Dict[str, Any]):
        """Start a Python bot process"""
        python_file = os.path.join(self.bot_folder_path, bot_config['python_file'])
        requirements_file = os.path.join(self.bot_folder_path, bot_config.get('requirements_file', ''))
        
        # Install requirements if they exist
        if os.path.exists(requirements_file):
            import subprocess
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
            except subprocess.CalledProcessError:
                print(f"Warning: Failed to install requirements for {bot_config['name']}")
        
        # Create pipe for communication
        parent_conn, child_conn = mp.Pipe()
        
        # Start bot process
        self.bot_process = mp.Process(
            target=self._run_python_bot,
            args=(child_conn, python_file, self.team),
            daemon=True
        )
        self.bot_process.start()
        
        # Save parent connection
        self.conn = parent_conn
    
    def _run_python_bot(self, conn, python_file: str, team: int):
        """Function that runs in the bot process"""
        try:
            # Add bot's directory to Python path
            sys.path.insert(0, os.path.dirname(python_file))
            
            # Load the module
            import importlib.util
            spec = importlib.util.spec_from_file_location("bot_module", python_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the bot class
            bot_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and hasattr(attr, 'get_output'):
                    bot_class = attr
                    break
            
            if not bot_class:
                raise ValueError(f"Could not find bot class in {python_file}")
            
            # Initialize the bot
            bot = bot_class(team)
            
            # Main loop
            while True:
                try:
                    # Get packet from parent process
                    packet = conn.recv()
                    
                    # Check if termination signal
                    if isinstance(packet, dict) and packet.get('terminate', False):
                        break
                    
                    # Process the packet and get controller output
                    controller_output = bot.get_output(packet)
                    
                    # Send back to parent process
                    conn.send(controller_output)
                    
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error in bot process: {e}")
                    conn.send({'error': str(e)})
                    
        except Exception as e:
            print(f"Error initializing bot: {e}")
            conn.send({'error': str(e)})
        finally:
            conn.close()
    
    def _convert_gamestate_to_packet(self, game_state: GameState) -> Dict[str, Any]:
        """Convert RLGym GameState to RLBot GameTickPacket"""
        # Convert game info
        packet = {
            'game_info': {
                'seconds_elapsed': game_state.game_info.get('seconds_elapsed', 0),
                'game_time_remaining': game_state.game_info.get('game_time_remaining', 300),
                'is_overtime': game_state.game_info.get('is_overtime', False),
                'is_round_active': game_state.game_info.get('is_round_active', True),
                'is_kickoff': game_state.game_info.get('is_kickoff', False),
                'world_gravity_z': game_state.game_info.get('world_gravity_z', -650),
                'game_speed': game_state.game_info.get('game_speed', 1.0)
            },
            'ball': {
                'physics': {
                    'location': game_state.ball.position.tolist(),
                    'rotation': [0, 0, 0],  # RLGym doesn't provide ball rotation
                    'velocity': game_state.ball.linear_velocity.tolist(),
                    'angular_velocity': game_state.ball.angular_velocity.tolist()
                }
            },
            'cars': []
        }
        
        # Add all cars to the packet
        for car_id, car in game_state.cars.items():
            car_data = {
                'physics': {
                    'location': car.position.tolist(),
                    'rotation': car.rotation.tolist() if hasattr(car, 'rotation') else [0, 0, 0],
                    'velocity': car.linear_velocity.tolist(),
                    'angular_velocity': car.angular_velocity.tolist()
                },
                'boost_amount': car.boost_amount,
                'jumped': car.has_jumped if hasattr(car, 'has_jumped') else False,
                'double_jumped': car.has_double_jumped if hasattr(car, 'has_double_jumped') else False,
                'on_ground': car.on_ground if hasattr(car, 'on_ground') else True,
                'supersonic': car.is_super_sonic if hasattr(car, 'is_super_sonic') else False,
                'demolished': car.is_demolished if hasattr(car, 'is_demolished') else False,
                'has_wheel_contact': car.has_wheel_contact if hasattr(car, 'has_wheel_contact') else True,
                'team': car.team_num
            }
            packet['cars'].append(car_data)
        
        return packet
    
    def _convert_controller_to_action(self, controller_inputs: Dict[str, float]) -> np.ndarray:
        """Convert RLBot controller inputs to RLGym action format"""
        action = np.zeros(8)
        
        # Map controller inputs to action array
        # RLGym action space: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        action[0] = controller_inputs.get('throttle', 0)
        action[1] = controller_inputs.get('steer', 0)
        action[2] = controller_inputs.get('pitch', 0)
        action[3] = controller_inputs.get('yaw', 0)
        action[4] = controller_inputs.get('roll', 0)
        action[5] = float(controller_inputs.get('jump', False))
        action[6] = float(controller_inputs.get('boost', False))
        action[7] = float(controller_inputs.get('handbrake', False))
        
        return action