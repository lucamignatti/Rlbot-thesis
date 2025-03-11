# Final 2v2 Rocket League Curriculum Design

After careful consideration of skill progression, achievability, and 2v2 team dynamics, I present this finalized curriculum design for developing capable Rocket League agents.

## Stage 1: Movement Foundations

### Base Task: Directional Control
- **State Mutator**: 
  - `FixedTeamSizeMutator(blue_size=1, orange_size=0)`
  - `CarPositionMutator` (random starting positions)
  - `TargetPositionMutator` (stationary targets)
- **Reward Function**: 
  - `ParameterizedReward` for target proximity (dispersion=1.0)
  - `VelocityReward` for maintaining speed
  - `StabilityReward` for controlled movement
- **Termination Condition**: Target reached or out-of-bounds
- **Truncation Condition**: `TimeoutCondition(8)`
- **Progression Requirements**:
  - `min_success_rate=0.85` 
  - `min_episodes=50`

### Skill Modules:
1. **Basic Driving**
   - **Focus**: Driving efficiently to targets
   - **Progression**: From simple straight lines to curved paths

2. **Boost Collection**
   - **Focus**: Collecting boost while maintaining momentum
   - **Progression**: Increasing complexity of boost pad layouts

3. **Basic Maneuvers**
   - **Focus**: Powerslides, half-flips, wave-dashes
   - **Progression**: From isolated mechanics to combined movements

## Stage 2: Ball Engagement

### Base Task: First Touch
- **State Mutator**: 
  - `FixedTeamSizeMutator(blue_size=1, orange_size=0)`
  - `BallPositionMutator` (stationary ball, various distances)
  - `CarPositionMutator` (varied approach angles)
- **Reward Function**: 
  - `KRCReward` combining:
    - `BallProximityReward(dispersion=0.8)` (approach)
    - `TouchBallReward(weight=0.3)` (contact)
    - `TouchQualityReward(weight=0.5)` (solid vs. glancing)
- **Termination Condition**: Ball touched or out-of-bounds
- **Truncation Condition**: `TimeoutCondition(6)`
- **Progression Requirements**:
  - `min_success_rate=0.8`
  - `min_episodes=75`

### Skill Modules:
1. **Approach Paths**
   - **Focus**: Approaching from various angles
   - **Progression**: From direct to angled approaches

2. **Contact Quality**
   - **Focus**: Making solid vs. glancing touches
   - **Progression**: Requiring more centered contact

3. **Moving Ball Intercepts**
   - **Focus**: Timing contact with rolling balls
   - **Progression**: From slow to faster ball speeds

## Stage 3: Ball Control & Direction

### Base Task: Directed Touch
- **State Mutator**: 
  - `FixedTeamSizeMutator(blue_size=1, orange_size=0)`
  - `BallPositionMutator` (varied starting positions)
  - `CarPositionMutator` (positioned for approach)
  - `TargetZoneMutator` (areas to direct ball toward)
- **Reward Function**: 
  - `KRCReward` combining:
    - `TouchBallReward(weight=0.2)`
    - `BallVelocityDirectionReward(weight=0.6)` (toward target)
    - `BallControlDurationReward(weight=0.3)` (maintaining influence)
- **Termination Condition**: Ball in target zone or out-of-bounds
- **Truncation Condition**: `TimeoutCondition(10)` + `NoTouchTimeoutCondition(4)`
- **Progression Requirements**:
  - `min_success_rate=0.65`
  - `min_episodes=100`

### Skill Modules:
1. **Power Control**
   - **Focus**: Controlling hit strength
   - **Progression**: Various target velocities

2. **Consecutive Touches**
   - **Focus**: Making multiple controlled touches
   - **Progression**: Increasing number of required touches

3. **Ball Stopping**
   - **Focus**: Slowing and controlling moving balls
   - **Progression**: From slow to fast incoming balls

## Stage 4: Shooting Fundamentals

### Base Task: Goal Scoring
- **State Mutator**: 
  - `FixedTeamSizeMutator(blue_size=1, orange_size=0)`
  - `BallTowardGoalSpawnMutator(distance_from_goal=0.75)`
  - `CarPositionMutator` (behind ball, facing goal)
- **Reward Function**: 
  - `KRCReward` combining:
    - `BallVelocityToGoalReward(weight=0.7)`
    - `TouchBallToGoalAccelerationReward(weight=0.3)`
    - `create_distance_weighted_alignment_reward()`
- **Termination Condition**: `GoalCondition()` or ball stopped
- **Truncation Condition**: `TimeoutCondition(8)`
- **Progression Requirements**:
  - `min_success_rate=0.6`
  - `min_episodes=125`

### Skill Modules:
1. **Center Shots**
   - **Focus**: Direct shots from central positions
   - **Progression**: Increasing distance from goal

2. **Angled Shots**
   - **Focus**: Shots from progressively wider angles
   - **Progression**: More extreme angles, requiring redirection

3. **Rolling Shot Timing**
   - **Focus**: Timing shots on rolling balls
   - **Progression**: Faster rolling speeds, bouncing balls

## Stage 5: Wall & Air Mechanics

### Base Task: Wall Driving & Basic Aerials
- **State Mutator**: 
  - `FixedTeamSizeMutator(blue_size=1, orange_size=0)`
  - `BallPositionMutator` (low aerials and wall positions)
  - `CarPositionMutator` (ground or wall starting positions)
  - `CarBoostMutator(boost_amount=50)`
- **Reward Function**: 
  - `KRCReward` combining:
    - `TouchBallReward(weight=0.2)`
    - `WallDrivingReward(weight=0.3)` (for wall segments)
    - `AerialTouchHeightReward(weight=0.5)` (for aerial segments)
- **Termination Condition**: Ball touched or grounded
- **Truncation Condition**: `TimeoutCondition(8)`
- **Progression Requirements**:
  - `min_success_rate=0.55`
  - `min_episodes=150`

### Skill Modules:
1. **Wall Driving**
   - **Focus**: Controlling car on walls
   - **Progression**: From simple paths to curved wall routes

2. **Wall-to-Ground Transitions**
   - **Focus**: Smoothly transitioning between surfaces
   - **Progression**: Various transition points and angles

3. **Jump Aerials**
   - **Focus**: Basic aerial control (200-400 units height)
   - **Progression**: From stationary to moving aerial targets

## Stage 6: Beginning Team Play (2v0)

### Base Task: Teammate Awareness
- **State Mutator**: 
  - `FixedTeamSizeMutator(blue_size=2, orange_size=0)`
  - `BallPositionMutator` (neutral field positions)
  - `TeamPositionMutator` (varied formations)
- **Reward Function**: 
  - `KRCReward` combining:
    - `TeamSpacingReward(weight=0.7)` (optimal distance)
    - `TeamPossessionReward(weight=0.5)` (ball control)
    - `create_offensive_potential_reward()`
- **Termination Condition**: `GoalCondition()` or possession loss
- **Truncation Condition**: `TimeoutCondition(15)`
- **Progression Requirements**:
  - `min_success_rate=0.55`
  - `min_episodes=125`

### Skill Modules:
1. **Spacing Awareness**
   - **Focus**: Maintaining optimal distance from teammate
   - **Progression**: From static to dynamic teammate movement

2. **Support Positioning**
   - **Focus**: Positioning when teammate has possession
   - **Progression**: More complex support patterns

3. **Basic Passing Lanes**
   - **Focus**: Creating and identifying passing opportunities
   - **Progression**: From obvious to subtle passing lanes

## Stage 7: Defense & Goal-line Saves

### Base Task: Shot Blocking
- **State Mutator**: 
  - `FixedTeamSizeMutator(blue_size=1, orange_size=1)`
  - `BallPositionMutator` (shots toward goal)
  - `CarPositionMutator` (defensive positions)
  - `ShotSpeedMutator` (varied shot power)
- **Reward Function**: 
  - `KRCReward` combining:
    - `BlockSuccessReward(weight=0.7)` 
    - `DefensivePositioningReward(weight=0.5)`
    - `BallClearanceReward(weight=0.4)` (distance after touch)
- **Termination Condition**: `GoalCondition()` (negative for conceded)
- **Truncation Condition**: `TimeoutCondition(10)`
- **Progression Requirements**:
  - `min_success_rate=0.6`
  - `min_episodes=150`
- **Bot Skill Range**: `{(0.2, 0.4): 1.0}` (basic shots)

### Skill Modules:
1. **Goal-line Saves**
   - **Focus**: Last-ditch saves directly on goal line
   - **Progression**: From central to corner shots

2. **Shadow Defense**
   - **Focus**: Backing up while facing the ball
   - **Progression**: From straight to curved shadow paths

3. **Recovery After Save**
   - **Focus**: Quick reorientation after save touch
   - **Progression**: From simple to more awkward landing angles

## Stage 8: Intermediate Ball Control

### Base Task: Close Control & Dribbling Setup
- **State Mutator**: 
  - `FixedTeamSizeMutator(blue_size=1, orange_size=0)`
  - `BallPositionMutator` (close to car)
  - `CarPositionMutator` (behind ball)
  - `CarBoostMutator(boost_amount=50)`
- **Reward Function**: 
  - `KRCReward` combining:
    - `BallProximityReward(dispersion=0.5)` (keeping ball close) 
    - `BallControlDurationReward(weight=0.6)` (time with control)
    - `ForwardProgressReward(weight=0.4)` (advancing with ball)
- **Termination Condition**: Ball control lost or goal scored
- **Truncation Condition**: `TimeoutCondition(12)`
- **Progression Requirements**:
  - `min_success_rate=0.5`
  - `min_episodes=175`

### Skill Modules:
1. **Close Following**
   - **Focus**: Keeping ball close while moving forward
   - **Progression**: Increasing speed and distance requirements

2. **Cut Control**
   - **Focus**: Changing direction while maintaining possession
   - **Progression**: From gentle to sharp directional changes

3. **Pop Setup**
   - **Focus**: Setting up ball for pop/flick
   - **Progression**: More precisely positioned setups

## Stage 9: 2v2 Defensive Rotation

### Base Task: Team Defense
- **State Mutator**: 
  - `FixedTeamSizeMutator(blue_size=2, orange_size=2)`
  - `BallPositionMutator` (attacking third for opponent)
  - `TeamPositionMutator` (defensive positions)
  - `OpponentPositionMutator` (attacking positions)
- **Reward Function**: 
  - `KRCReward` combining:
    - `TeamDefenseReward(weight=0.7)` (coordinated coverage)
    - `ChallengeSuccessReward(weight=0.5)` (winning 50/50s)
    - `GoalPreventionReward(weight=0.6)` (successful clearances)
- **Termination Condition**: `GoalCondition()` or cleared to safety
- **Truncation Condition**: `TimeoutCondition(20)`
- **Progression Requirements**:
  - `min_success_rate=0.55`
  - `min_episodes=175`
- **Bot Skill Range**: `{(0.3, 0.5): 1.0}`

### Skill Modules:
1. **First Man Challenge**
   - **Focus**: When and how to challenge as first defender
   - **Progression**: From obvious to nuanced challenge decisions

2. **Last Man Positioning**
   - **Focus**: Safe positioning as last defender
   - **Progression**: More threatening offensive scenarios

3. **Recovery Rotation**
   - **Focus**: Rotating back after challenge
   - **Progression**: Faster rotations, more pressure scenarios

## Stage 10: 2v2 Offensive Coordination

### Base Task: Team Offense
- **State Mutator**: 
  - `FixedTeamSizeMutator(blue_size=2, orange_size=2)`
  - `BallPositionMutator` (attacking third for agent team)
  - `TeamPositionMutator` (offensive positions)
  - `OpponentPositionMutator` (defensive positions)
- **Reward Function**: 
  - `KRCReward` combining:
    - `TeamPossessionReward(weight=0.5)`
    - `PassCompletionReward(weight=0.7)`
    - `ScoringOpportunityCreationReward(weight=0.6)`
    - `create_distance_weighted_alignment_reward()`
- **Termination Condition**: `GoalCondition()` or possession lost
- **Truncation Condition**: `TimeoutCondition(20)`
- **Progression Requirements**:
  - `min_success_rate=0.5`
  - `min_episodes=200`
- **Bot Skill Range**: `{(0.3, 0.5): 1.0}`

### Skill Modules:
1. **Pass Execution**
   - **Focus**: Directing ball to teammate
   - **Progression**: From simple to more complex passing patterns

2. **Receiving Passes**
   - **Focus**: Positioning to receive and follow up
   - **Progression**: From perfect to imperfect passes

3. **Pass-or-Shoot Decision**
   - **Focus**: Choosing between passing and shooting
   - **Progression**: More ambiguous decision scenarios

## Stage 11: Intermediate Aerials & Wall Play

### Base Task: Dynamic Aerial Control
- **State Mutator**: 
  - `FixedTeamSizeMutator(blue_size=1, orange_size=0)`
  - `BallPositionMutator` (moderate aerial heights, 400-800 units)
  - `BallVelocityMutator` (moving aerial targets)
  - `CarPositionMutator` (varied takeoff positions)
  - `CarBoostMutator(boost_amount=75)`
- **Reward Function**: 
  - `KRCReward` combining:
    - `TouchBallReward(weight=0.2)`
    - `AerialControlReward(weight=0.4)` (stability in air)
    - `AerialDirectionalTouchReward(weight=0.6)` (controlled redirects)
- **Termination Condition**: Ball touched or grounded
- **Truncation Condition**: `TimeoutCondition(10)`
- **Progression Requirements**:
  - `min_success_rate=0.45`
  - `min_episodes=200`

### Skill Modules:
1. **Fast Aerials**
   - **Focus**: Quick takeoff with boost+jump combo
   - **Progression**: Higher, faster aerial targets

2. **Wall-to-Air**
   - **Focus**: Transitioning from wall to aerial play
   - **Progression**: Various wall positions and jump angles

3. **Air Roll Control**
   - **Focus**: Basic aerial car rotation
   - **Progression**: From simple to more complex rotations

## Stage 12: Full 2v2 Integration

### Base Task: Complete 2v2 Matches
- **State Mutator**: 
  - `FixedTeamSizeMutator(blue_size=2, orange_size=2)`
  - `KickoffMutator()` (standard kickoffs)
- **Reward Function**: 
  - `LucySKGReward` (full team-oriented reward function)
  - Supplemented with:
    - Win probability component
    - Team synergy metrics
- **Termination Condition**: `GoalCondition()`
- **Truncation Condition**: `TimeoutCondition(300)` (full matches)
- **Progression Requirements**:
  - `min_win_rate=0.55` against comparable bot difficulty
  - `min_episodes=250`
- **Bot Skill Range**: `{(0.4, 0.6): 0.7, (0.6, 0.7): 0.3}`

### Skill Modules:
1. **Kickoff Strategy**
   - **Focus**: Team coordination at kickoff
   - **Progression**: More varied kickoff positions and strategies

2. **Full-field Rotation**
   - **Focus**: Seamless rotation across entire field
   - **Progression**: Faster gameplay requiring quicker decisions

3. **Game State Adaptation**
   - **Focus**: Adjusting to score, time, and boost states
   - **Progression**: Various game situations requiring strategic adjustments

## Implementation Strategy

### Skill Progression Framework
- Each stage uses approximately 70% success criteria before advancement
- Dynamic difficulty adjustment based on rolling performance window
- Interleaved training between stages to prevent catastrophic forgetting
- Performance-based rehearsal of previously learned skills (15-20% of training time)

### Training Architecture
- Synchronized training of both team agents
- Shared reward signals for team objectives
- Individual agent rewards for position-specific behaviors
- Alternating role assignments to develop flexible playstyles

### Evaluation Framework
- Regular evaluation matches against static bot difficulties
- Performance tracking across skill-specific metrics
- Win rate progression requirements increasing with curriculum stage
- Dedicated evaluation scenarios testing specific skill combinations

This curriculum design provides a systematic approach to developing 2v2 Rocket League agents, with each stage building incrementally on previous capabilities. The focus on achievable skill progression ensures the agent can master fundamental mechanics before attempting more complex behaviors, while the 2v2-specific elements develop proper team coordination from an early stage.