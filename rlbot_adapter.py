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
    """Adapter that allows RLGym to use bots from the RLBot framework."""
    
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
        self.is_rlgym_bot = False
        self.tick_skip = 8  # Default, will be updated from bot config
        
    def start(self):
        """Start the bot process and establish communication"""
        cfg_file = self._find_bot_cfg_file()
        if not cfg_file:
            raise ValueError(f"Could not find cfg file for bot {self.bot_id}")
            
        bot_config = self._parse_cfg_file(cfg_file)
        
        # Check if this is an RLGym bot by looking for indicators
        self.is_rlgym_bot = self._is_rlgym_bot(bot_config)
        if self.is_rlgym_bot:
            print(f"Detected RLGym bot: {self.bot_id}")
            self.tick_skip = self._get_tick_skip(bot_config)
        
        self._start_python_bot(bot_config)
        self.is_running = True

    def _is_rlgym_bot(self, bot_config: Dict[str, Any]) -> bool:
        """Check if this is an RLGym-based bot by examining code patterns"""
        python_file = bot_config.get('python_file', '')
        if not python_file or not os.path.exists(python_file):
            return False
            
        try:
            with open(python_file, 'r') as f:
                code = f.read()
                # Look for common RLGym imports and patterns
                rlgym_indicators = [
                    'rlgym_compat',
                    'AdvancedObs',
                    'GameState',
                    'PhysicsObject',
                    'tick_skip',
                    'self.obs_builder',
                    'self.act_parser'
                ]
                return any(indicator in code for indicator in rlgym_indicators)
        except:
            return False

    def _get_tick_skip(self, bot_config: Dict[str, Any]) -> int:
        """Get the tick skip value from an RLGym bot's code"""
        python_file = bot_config.get('python_file', '')
        if not python_file or not os.path.exists(python_file):
            return 8
            
        try:
            with open(python_file, 'r') as f:
                code = f.read()
                # Look for tick_skip assignment
                if 'self.tick_skip = ' in code:
                    # Try to extract the value
                    import re
                    match = re.search(r'self\.tick_skip\s*=\s*(\d+)', code)
                    if match:
                        return int(match.group(1))
        except:
            pass
        return 8  # Default value

    def get_action(self, game_state: GameState) -> np.ndarray:
        """Get the next action from the bot for the current game state"""
        if not self.is_running:
            print(f"Bot {self.bot_id} is not running")
            return np.zeros(8)  # Return no-op action
            
        try:
            # Convert GameState to packet format
            packet = self._convert_gamestate_to_packet(game_state)
            
            # Send packet to bot process
            self.conn.send(packet)
            
            # Get response from bot
            controller_inputs = self.conn.recv()
            
            # Convert controller inputs to RLGym action format
            action = self._convert_controller_to_action(controller_inputs)
            
            if self.is_rlgym_bot:
                # RLGym bots expect continuous actions in [-1, 1]
                action = np.clip(action, -1, 1)
                
            return action
            
        except Exception as e:
            print(f"Error getting action from bot: {e}")
            return np.zeros(8)  # Return no-op action

    def _convert_gamestate_to_packet(self, game_state: GameState) -> Dict[str, Any]:
        """Convert RLGym GameState to RLBot GameTickPacket format"""
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
                'team': car.team_num,
                'name': f'Bot {car_id}',
                'index': car_id
            }
            packet['cars'].append(car_data)
            
        return packet

    def _convert_controller_to_action(self, controller_inputs: Dict[str, float]) -> np.ndarray:
        """Convert RLBot controller inputs to RLGym action format"""
        action = np.zeros(8)
        
        # Map controller inputs to action array based on bot type
        if self.is_rlgym_bot:
            # RLGym bots use continuous actions in [-1, 1]
            action[0] = controller_inputs.get('throttle', 0)  # throttle
            action[1] = controller_inputs.get('steer', 0)    # steer
            action[2] = controller_inputs.get('pitch', 0)    # pitch
            action[3] = controller_inputs.get('yaw', 0)      # yaw
            action[4] = controller_inputs.get('roll', 0)     # roll
            action[5] = float(controller_inputs.get('jump', False))    # jump
            action[6] = float(controller_inputs.get('boost', False))   # boost
            action[7] = float(controller_inputs.get('handbrake', False)) # handbrake
        else:
            # Regular RLBot format
            action[0] = controller_inputs.get('throttle', 0)
            action[1] = controller_inputs.get('steer', 0)
            action[2] = controller_inputs.get('pitch', 0)
            action[3] = controller_inputs.get('yaw', 0)
            action[4] = controller_inputs.get('roll', 0)
            action[5] = float(controller_inputs.get('jump', False))
            action[6] = float(controller_inputs.get('boost', False))
            action[7] = float(controller_inputs.get('handbrake', False))
        
        return action

    def stop(self):
        """Stop the bot process and clean up resources"""
        if self.bot_process and self.bot_process.is_alive():
            self.bot_process.terminate()
            self.bot_process.join()
        if self.conn:
            self.conn.close()
        self.is_running = False

    def _find_bot_cfg_file(self) -> Optional[str]:
        """Find the bot's cfg file in the bot folder"""
        for root, _, files in os.walk(self.bot_folder_path):
            for file in files:
                if file.endswith('.cfg'):
                    return os.path.join(root, file)
        return None

    def _parse_cfg_file(self, cfg_file: str) -> Dict[str, Any]:
        """Parse the bot cfg file to extract important information"""
        config = configparser.ConfigParser()
        config.read(cfg_file)
        
        bot_config = {}
        if 'Locations' in config:
            python_file = config['Locations'].get('python_file', '')
            if python_file:
                if not os.path.isabs(python_file):
                    python_file = os.path.join(os.path.dirname(cfg_file), python_file)
                bot_config['python_file'] = python_file
            
            requirements = config['Locations'].get('requirements', '')
            if requirements and not os.path.isabs(requirements):
                requirements = os.path.join(os.path.dirname(cfg_file), requirements)
            bot_config['requirements_file'] = requirements
            
        if 'Details' in config:
            bot_config['name'] = config['Details'].get('name', 'Unknown')
            bot_config['language'] = config['Details'].get('language', 'python')
            
        return bot_config

    def _start_python_bot(self, bot_config: Dict[str, Any]):
        """Start a Python bot process"""
        if not bot_config.get('python_file'):
            raise ValueError(f"No Python file specified for bot {self.bot_id}")
            
        # Create pipe for communication
        parent_conn, child_conn = mp.Pipe()
        self.conn = parent_conn
        
        # Start bot process
        self.bot_process = mp.Process(
            target=self._run_python_bot,
            args=(child_conn, bot_config['python_file'], self.team)
        )
        self.bot_process.start()

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
            if hasattr(bot, 'initialize_agent'):
                bot.initialize_agent()
            
            # Main loop
            while True:
                try:
                    # Get packet from parent process
                    packet = conn.recv()
                    
                    # Get bot's move
                    controller = bot.get_output(packet)
                    
                    # Send response back to parent
                    conn.send(controller)
                    
                except EOFError:
                    break  # Parent process closed connection
                except Exception as e:
                    print(f"Error in bot process: {e}")
                    conn.send({})  # Send empty controller state
                    
        except Exception as e:
            print(f"Failed to start bot: {e}")
        finally:
            conn.close()