import os
import random
import configparser
from typing import Dict, List, Any, Tuple, Optional
from rlbot_adapter import RLBotAdapter

class RLBotPackRegistry:
    """Registry for bots from the RLBotPack."""
    def __init__(self, rlbotpack_path: str):
        """
        Initialize the registry with path to RLBotPack.
        
        Args:
            rlbotpack_path: Path to the RLBotPack repository
        """
        self.rlbotpack_path = rlbotpack_path
        self.available_bots: Dict[str, Dict[str, Any]] = {}
        self.loaded_bot_adapters: Dict[str, RLBotAdapter] = {}
        self.force_bot: Optional[str] = None  # For forcing a specific bot
        
        # Scan the RLBotPack folder to find available bots
        self._scan_rlbotpack()
    
    def _scan_rlbotpack(self):
        """Scan the RLBotPack directory to find available bots"""
        # Scan RLBotPack/RLBotPack directory
        pack_dir = os.path.join(self.rlbotpack_path, "RLBotPack")
        if not os.path.isdir(pack_dir):
            print(f"Warning: {pack_dir} not found")
            return
        
        # First level contains categories (like HiveBot, ReliefBot, etc.)
        for category in os.listdir(pack_dir):
            category_path = os.path.join(pack_dir, category)
            if not os.path.isdir(category_path):
                continue
                
            # Each category may contain one or more bot folders
            for bot_folder in os.listdir(category_path):
                bot_path = os.path.join(category_path, bot_folder)
                if not os.path.isdir(bot_path):
                    continue
                
                # Try to find a cfg file in this folder
                cfg_files = [f for f in os.listdir(bot_path) if f.endswith('.cfg')]
                if not cfg_files:
                    continue
                
                # Use the first cfg file found
                cfg_file = os.path.join(bot_path, cfg_files[0])
                
                try:
                    # Extract bot info from cfg file
                    bot_info = self._extract_bot_info(cfg_file)
                    
                    # Add to available bots dictionary
                    bot_id = f"{category}/{bot_folder}"
                    self.available_bots[bot_id] = {
                        'id': bot_id,
                        'name': bot_info.get('name', bot_folder),
                        'path': bot_path,
                        'language': bot_info.get('language', 'unknown'),
                        'skill_estimate': bot_info.get('skill_estimate', 0.5),
                        'tags': bot_info.get('tags', [])
                    }
                    
                except Exception as e:
                    print(f"Error processing bot {bot_path}: {str(e)}")
    
    def _extract_bot_info(self, cfg_file: str) -> Dict[str, Any]:
        """Extract bot information from cfg file"""
        config = configparser.ConfigParser()
        config.read(cfg_file)
        
        bot_info = {}
        if 'Details' in config:
            bot_info['name'] = config['Details'].get('name', '')
            bot_info['developer'] = config['Details'].get('developer', '')
            bot_info['description'] = config['Details'].get('description', '')
            bot_info['language'] = config['Details'].get('language', 'python')
            bot_info['tags'] = [tag.strip() for tag in config['Details'].get('tags', '').split(',') if tag.strip()]
            
        # Estimate skill level based on tags or other info
        bot_info['skill_estimate'] = self._estimate_skill_level(bot_info)
        
        return bot_info
    
    def _estimate_skill_level(self, bot_info: Dict[str, Any]) -> float:
        """Estimate bot skill level based on available information"""
        skill = 0.5  # Default mid-level
        
        # Check tags for skill indicators
        tags = [tag.lower() for tag in bot_info.get('tags', [])]
        
        if any(tag in tags for tag in ['beginner', 'easy', 'simple']):
            skill = 0.2
        elif any(tag in tags for tag in ['intermediate']):
            skill = 0.5
        elif any(tag in tags for tag in ['advanced', 'expert']):
            skill = 0.8
        elif any(tag in tags for tag in ['grand champion', 'gc']):
            skill = 0.9
        elif any(tag in tags for tag in ['pro', 'competitive']):
            skill = 0.85
        
        # Clamp to valid range
        return min(1.0, max(0.1, skill))
    
    def get_available_bots(self, min_skill: float = 0.0, max_skill: float = 1.0,
                          tags: Optional[List[str]] = None, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of available bots filtered by criteria.
        
        Args:
            min_skill: Minimum skill level (0.0-1.0)
            max_skill: Maximum skill level (0.0-1.0)
            tags: Filter by these tags
            language: Filter by programming language
            
        Returns:
            List of bot dictionaries matching criteria
        """
        filtered_bots = []
        
        for bot_info in self.available_bots.values():
            # Apply skill filter
            if bot_info['skill_estimate'] < min_skill or bot_info['skill_estimate'] > max_skill:
                continue
                
            # Apply language filter
            if language and bot_info['language'].lower() != language.lower():
                continue
                
            # Apply tag filter
            if tags:
                bot_tags = set(t.lower() for t in bot_info.get('tags', []))
                if not all(tag.lower() in bot_tags for tag in tags):
                    continue
                
            filtered_bots.append(bot_info)
        
        return filtered_bots
    
    def create_bot_adapter(self, bot_id: str, team: int = 1) -> RLBotAdapter:
        """
        Create an adapter for a specific bot.
        
        Args:
            bot_id: ID of the bot to load
            team: Team number (0=blue, 1=orange)
            
        Returns:
            RLBotAdapter instance
        """
        if self.force_bot and self.force_bot != bot_id:
            # If a specific bot is forced, use that instead
            bot_id = self.force_bot
            
        if bot_id not in self.available_bots:
            raise ValueError(f"Bot with ID {bot_id} not found")
        
        bot_info = self.available_bots[bot_id]
        
        # Create a cached adapter if it doesn't exist
        cache_key = f"{bot_id}_{team}"
        if cache_key not in self.loaded_bot_adapters:
            adapter = RLBotAdapter(bot_info['path'], team)
            self.loaded_bot_adapters[cache_key] = adapter
        
        return self.loaded_bot_adapters[cache_key]
    
    def get_random_bot(self, min_skill: float = 0.0, max_skill: float = 1.0,
                      tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get a random bot adapter based on criteria.
        
        Args:
            min_skill: Minimum skill level (0.0-1.0)
            max_skill: Maximum skill level (0.0-1.0)
            tags: Required tags for the bot
            
        Returns:
            Bot info dictionary
        """
        bots = self.get_available_bots(min_skill, max_skill, tags)
        
        if not bots:
            raise ValueError(
                f"No bots found matching criteria (skill: {min_skill}-{max_skill}, "
                f"tags: {tags if tags else 'any'})"
            )
        
        # Select random bot
        return random.choice(bots)
    
    def cleanup(self):
        """Stop all bot processes and clean up resources"""
        for adapter in self.loaded_bot_adapters.values():
            try:
                adapter.stop()
            except:
                pass
        self.loaded_bot_adapters.clear()