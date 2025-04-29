### LETS DO IT
- Mission
    - Train robots to operate in a DnD world (ex: get a treasure, navigate the maze)
    - Make it a playable game where the players can use reward functions to train the robot adventurers
        - Teach how AI works
    - Learn RL, robotics, and sharpen programming skills
    - Get internships/jobs at a robotics company
### Phases
0. Setup and tutorials
1. Hello world - Done
    - Unity level editor with usd export (maze.usd)
    - Import maze.usd with physics to isaac lab
    - Recreate IsaacLab tutorials
2. Self contained project - Done
    - Isaac lab prims with maze.usd
    - Isaac lab cartpole with maze.usd
3. Direct learning - Done
    - Maze generation in unity
    - Robot with maze.usd and physics
    - Each robot with own maze.usd
4. Train robot to complete simple task
    - Maze.usd with goal point (chest) - Done
    - Robot runs to chest
        - Adding manual waypoints (using leatherback) - Done
        - Making the chest a waypoint using environment ID - Done
            - Get humanoid moving
            - Train a new policy using a robot with a policy assigned to it
            - 
        - Lula kinematics solver - Not useful
        - Cumotion - Not useful
5. Maze solver AI
    - Using isaacsim random maze generator
    - Figure out how to reference completed AI training while training other AI (policy)
        - Pre-trained walking humanoid
        - maze solver AI
        - Either AI controller or update policy
    - Make the ai
        - CuOpt?
            - https://build.nvidia.com/nvidia/nvidia-cuopt
        - Fleetmanager?
    - Plug in the maze solver to robot AI with solution to maze
    - Add rewards along the path it makes
6. Custom tasks
    - Robot avoids enemy
    - Enemy chases robot
    - Combat?
    - Team of robots coordinating to reach goal
7. Third AI (co-op humanoids)
    - Fleetmanager type thing or overseeing eye
    - Faster work together training
8. Pre-train basic skills
    - Navigation
    - Locomotion
9. UI for player
10. DnD theme

----------------------------------------------
### Leaderboard

- Top pains so far
    - Finding solutions to errors online (resources in general)
        - Multiple environments broke physics on usd file
    - Learning the concepts
        - Hard to find relevant cfgs
        - Manipulating USD files per environment
        - Explaining each training algorithym (ex: SB3, Rl_games)
    - Setting up project
        - Project folder needed to be moved to direct
        - Collecting robots from composer
        - vscode git ignore, ignores usd files
- Top joys
    - Duplicating an environment that has a maze and a robot
    - Exporting first usd scene to IsaacLab
    - 2.0 IsaacLab release
        - Improved tutorials
        - Folder organization
        - Git hub repo links for documentation
    - Setting up VScode launch.json
- Top future pains
    - Multiple AIs working together
    - Random maze generation for each environment
    - Complex custom tasks
    - Using the training data from IsaacLab in a game
        - Putting environment into IsaacLab
        - or
        - Exporting the learning data from IsaacLab into another program
- Questions
    - Mapping animation to robots
    - Getting robots to move to specific location - Possible solution
        - Torch tensors (2/12 meeting)
----------------------------------------------
### Questions
