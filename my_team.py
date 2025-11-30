import random
import util
import time
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='AlphaBetaAgent', second='AlphaBetaAgent', num_training=0):
    """
    This function returns a list of two agents that will form the
    team, initialized using first_index and second_index as their agent
    index numbers.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class AlphaBetaAgent(CaptureAgent):

    def register_initial_state(self, game_state):
        self.starting_pos = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
        self.is_team_1 = self.index in game_state.get_red_team_indices()
        map_width = game_state.data.layout.width
        
        if self.is_team_1:
            self.map_boundary = (map_width // 2) - 1
        else:
            self.map_boundary = (map_width // 2)
        
        self.depth = 2 

    def choose_action(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in legal_actions: 
            legal_actions.remove(Directions.STOP)

        best_score = -float('inf')
        best_move = random.choice(legal_actions)
        
        alpha = -float('inf')
        beta = float('inf')

        for a in legal_actions:
            successor_game_state = game_state.generate_successor(self.index, a)
            score = self.alpha_beta(successor_game_state, 1, self.index, False, alpha, beta)
            
            if score > best_score:
                best_score = score
                best_move = a
            
            alpha = max(alpha, best_score)
        
        return best_move

    def alpha_beta(self, game_state, current_depth, current_agent, is_my_turn, alpha, beta):
        if current_depth == 0 or game_state.is_over():
            return self.calculate_heuristic_score(game_state)

        num_agents = game_state.get_num_agents()
        next_agent = (current_agent + 1) % num_agents
        
        if next_agent == self.index:
            next_depth = current_depth - 1
        else:
            next_depth = current_depth
        
        if next_depth < 0:
            return self.calculate_heuristic_score(game_state)
        
        if game_state.get_agent_state(next_agent).configuration is None:
            return self.alpha_beta(game_state, next_depth, next_agent, is_my_turn, alpha, beta)

        legal_moves = game_state.get_legal_actions(next_agent)
        if not legal_moves: 
            return self.calculate_heuristic_score(game_state)

        if is_my_turn:
            best_score = -float('inf')
            for a in legal_moves:
                successor_game_state = game_state.generate_successor(next_agent, a)

                next_next_is_enemy = ((next_agent + 1) % num_agents not in self.get_team(game_state))
                
                score = self.alpha_beta(successor_game_state, next_depth, next_agent, not next_next_is_enemy, alpha, beta)
                
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                
                if beta <= alpha: 
                    break 
            return best_score
        else:
            best_score = float('inf')
            for a in legal_moves:
                successor_game_state = game_state.generate_successor(next_agent, a)
                next_next_is_enemy = ((next_agent + 1) % num_agents not in self.get_team(game_state))
                
                score = self.alpha_beta(successor_game_state, next_depth, next_agent, not next_next_is_enemy, alpha, beta)
                
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha: 
                    break 
            return best_score

    def calculate_heuristic_score(self, game_state):
        
        agent_state = game_state.get_agent_state(self.index)
        agent_pos = agent_state.get_position()
        
        if agent_pos is None: return -10000

        current_score = game_state.get_score()
        if self.is_team_1: 
            winning = current_score > 0
        else: 
            winning = current_score < 0
            

        if winning:
            current_role = 'D' 
        else:
            team_indices = self.get_team(game_state)
            teammate_index = [i for i in team_indices if i != self.index][0]
            teammate_state = game_state.get_agent_state(teammate_index)
            teammate_pos = teammate_state.get_position()
            
            if not teammate_pos:
                current_role = 'O'
            else:
                map_w = game_state.data.layout.width
                if self.is_team_1:
                    my_progress = agent_pos[0]
                    mate_progress = teammate_pos[0]
                else:
                    my_progress = map_w - agent_pos[0]
                    mate_progress = map_w - teammate_pos[0]

                if my_progress > mate_progress:
                    current_role = 'O'
                elif my_progress < mate_progress: 
                    current_role = 'D'
                else: 
                    if self.index < teammate_index:
                        current_role = 'O'
                    else:
                        current_role = 'D'


        if self.is_team_1:
            heuristic_score = current_score
        else:
            heuristic_score = -current_score
        heuristic_score *= 100000 
        
        food_list = self.get_food(game_state).as_list()
        opponents = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        dist_to_base = self.get_distance_to_boundary(game_state, agent_pos)

        
        if current_role == 'O':
            
            threats = [a for a in opponents if not a.is_pacman and a.get_position() and a.scared_timer < 3]
            dist_closest_threat = 9999
            if threats:
                dist_closest_threat = min([self.get_maze_distance(agent_pos, g.get_position()) for g in threats])

            if dist_closest_threat <= 1: 
                return -float('inf') 
            
            if dist_closest_threat <= 2: 
                heuristic_score -= 20000 / (dist_closest_threat + 0.1)

            needs_return = False
            
            if agent_state.num_carrying >= 3:
                needs_return = True
                
            elif len(food_list) <= 2:
                needs_return = True
                
            elif agent_state.num_carrying > 0 and game_state.data.timeleft < 100:
                needs_return = True
                
            elif agent_state.num_carrying > 0 and dist_closest_threat <= 5:
                needs_return = True

            if needs_return:
                heuristic_score += 50000 
                heuristic_score -= dist_to_base * 500 
            else:
                if len(food_list) > 0:
                    dist_closest_food = min([self.get_maze_distance(agent_pos, f) for f in food_list])
                    heuristic_score -= dist_closest_food * 2
                    heuristic_score -= len(food_list) * 50

        else:
            
            if agent_state.is_pacman:
                heuristic_score -= 10000 
                
            invaders = [a for a in opponents if a.get_position() and a.is_pacman]
            
            if agent_state.scared_timer > 0:
                if invaders:
                    dists = [self.get_maze_distance(agent_pos, a.get_position()) for a in invaders]
                    heuristic_score += min(dists) * 1000 
                else:
                    heuristic_score -= dist_to_base 
                return heuristic_score

            if len(invaders) > 0:
                dists = [self.get_maze_distance(agent_pos, a.get_position()) for a in invaders]
                min_dist = min(dists)
                heuristic_score -= min_dist * 1000 
            else:
                heuristic_score -= dist_to_base 

        return heuristic_score

    def get_distance_to_boundary(self, game_state, current_pos):
        target_x = self.map_boundary
        height = game_state.data.layout.height
        boundary_spaces = []
        for y in range(1, height - 1):
             if not game_state.has_wall(target_x, y):
                 boundary_spaces.append((target_x, y))
        if not boundary_spaces: return 9999
        return min([self.get_maze_distance(current_pos, t) for t in boundary_spaces])