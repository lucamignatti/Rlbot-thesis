"""Registry for RLBotPack bots."""
import os
import random
from typing import Dict, List, Any, Tuple, Optional, Set
from .adapter import RLBotAdapter

class RLBotPackRegistry:
    """Registry for bots from the RLBotPack."""
    
    def __init__(self, rlbotpack_path: str):
        """Initialize the registry with path to RLBotPack."""
        self.rlbotpack_path = rlbotpack_path
        self.available_bots: Dict[str, Dict[str, Any]] = {}
        self.loaded_bot_adapters: Dict[str, RLBotAdapter] = {}
        self.force_bot: Optional[str] = None
        
        # Scan the RLBotPack folder
        self._scan_rlbotpack()
    
    def _scan_rlbotpack(self):
        """Scan the RLBotPack directory to find available bots."""
        pack_dir = os.path.join(self.rlbotpack_path, "RLBotPack")
        if not os.path.isdir(pack_dir):
            print(f"Warning: {pack_dir} not found")
            return
        
        # Scan categories (like HiveBot, ReliefBot, etc.)
        for category in os.listdir(pack_dir):
            category_path = os.path.join(pack_dir, category)
            if not os.path.isdir(category_path):
                continue
                
            # Each category may contain one or more bot folders
            for bot_folder in os.listdir(category_path):
                bot_path = os.path.join(category_path, bot_folder)
                if not os.path.isdir(bot_path):
                    continue
                
                # Look for Python files to identify bots
                python_files = []
                for root, _, files in os.walk(bot_path):
                    python_files.extend(
                        os.path.join(root, f) for f in files 
                        if f.endswith('.py')
                    )
                
                if not python_files:
                    continue
                    
                # Add to available bots dictionary
                bot_id = f"{category}/{bot_folder}"
                
                # Estimate bot skill based on naming and category
                skill_estimate = self._estimate_skill_level(category, bot_folder)
                
                # Determine tags from code analysis
                tags = self._analyze_bot_tags(python_files[0])
                
                self.available_bots[bot_id] = {
                    'id': bot_id,
                    'name': bot_folder,
                    'path': bot_path,
                    'skill_estimate': skill_estimate,
                    'tags': tags
                }
    
    def _estimate_skill_level(self, category: str, bot_name: str) -> float:
        """
        Estimate bot skill level based on category and name.
        Returns a value between 0.0 and 1.0.
        """
        # Known high-skill bots and categories
        high_skill_indicators = [
            'KarmikRL', 'AcrBot', 'ReliefBot', 'RLGym', 'NectoRL',
            'Nexto', 'Necto', 'NomBot', 'ApexRL', 'Bots/rlgym'
        ]
        
        # Known mid-skill bots and categories
        mid_skill_indicators = [
            'HiveBot', 'BotimusPrime', 'DragonBot', 'BeastBot',
            'PythonExample', 'Bots/utility'
        ]
        
        # Convert to lowercase for comparison
        category_lower = category.lower()
        name_lower = bot_name.lower()
        
        # Check for skill indicators
        if any(indicator.lower() in category_lower or indicator.lower() in name_lower 
               for indicator in high_skill_indicators):
            return random.uniform(0.8, 1.0)
            
        if any(indicator.lower() in category_lower or indicator.lower() in name_lower 
               for indicator in mid_skill_indicators):
            return random.uniform(0.5, 0.7)
            
        # Default to lower-mid skill range
        return random.uniform(0.3, 0.5)
    
    def _analyze_bot_tags(self, python_file: str) -> List[str]:
        """Analyze bot code to determine its characteristics."""
        tags = []
        try:
            with open(python_file, 'r') as f:
                code = f.read().lower()
                
                # Check for aerial capabilities
                if any(term in code for term in ['aerial', 'double_jump', 'air_dodge']):
                    tags.append('aerial')
                    
                # Check for dribbling capabilities
                if any(term in code for term in ['dribble', 'carry', 'flick']):
                    tags.append('dribbling')
                    
                # Check for defensive play
                if any(term in code for term in ['defend', 'save', 'clear_ball']):
                    tags.append('defensive')
                    
                # Check for boost management
                if any(term in code for term in ['boost_pickup', 'boost_pathing']):
                    tags.append('boost_aware')
                    
                # Check for team play
                if any(term in code for term in ['pass', 'teammate', 'team_play']):
                    tags.append('team_player')
                    
        except:
            pass
            
        return tags
    
    def get_available_bots(self, min_skill: float = 0.0, max_skill: float = 1.0,
                          tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get list of available bots filtered by criteria."""
        filtered_bots = []
        
        for bot_info in self.available_bots.values():
            # Apply skill filter
            if bot_info['skill_estimate'] < min_skill or bot_info['skill_estimate'] > max_skill:
                continue
                
            # Apply tag filter
            if tags:
                bot_tags = set(t.lower() for t in bot_info.get('tags', []))
                if not all(tag.lower() in bot_tags for tag in tags):
                    continue
                
            filtered_bots.append(bot_info)
        
        return filtered_bots
    
    def create_bot_adapter(self, bot_id: str, team: int = 1) -> RLBotAdapter:
        """Create an adapter for a specific bot."""
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
        """Get a random bot based on criteria."""
        bots = self.get_available_bots(min_skill, max_skill, tags)
        
        if not bots:
            raise ValueError(
                f"No bots found matching criteria (skill: {min_skill}-{max_skill}, "
                f"tags: {tags if tags else 'any'})"
            )
        
        return random.choice(bots)
    
    def cleanup(self):
        """Stop all bot processes and clean up resources."""
        for adapter in self.loaded_bot_adapters.values():
            try:
                adapter.stop()
            except:
                pass
        self.loaded_bot_adapters.clear()