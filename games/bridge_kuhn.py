import random
from itertools import product
from typing import List, Tuple, Union

import numpy as np

from games.base import base


class Bridge_Kuhn_Poker(base):
    curr_player = 0
    curr_bets = [0,0]
    cards = [0,0]
    winner = 0
    ended = False
    num_players = 2
    string_to_act = {"check": -1, "bet": -1, "left": 2, "right": 3, "jump": 4}
    def __init__(self, _num_players: int=2, _bridge_len:int=4):
        """Additional bridge crossing phase (rewards received at the end of episode):
            - Player only observes its own position on the bridge, and if its in waiting state (during mid turn all players crossed the bridge).
            - If both player jumps/go right/left/illegal action, both get reward 0. Illegal actions being bet and check.
            - If only one player jumps, they get -1, the other gets +1.
            - If only player goes right, they get +1, other agent (who went left) gets -1.
        When Kuhn poker playing phase starts:
            - distribute cards
            - Same rules as vanilla Kuhn poker, but if they perform illegal actions (left right jump), they receive -10.
            (Or 0 if both performs illegal action).
        Impose maxinum number of turns. If reaches max turn, end the game.
        
        Args:
            _num_players (int, optional): Number of players. Defaults to 2. Only supports 2 for now.
            _bridge_len (int, optional): Length of bridge crossing phase. Defaults to 10.
        """
        random.seed(1)
        self.num_players = _num_players
        self.bridge_len = _bridge_len
        self.max_turn_number = 2 * (self.bridge_len + 1)

        num_kuhn_states = (self.num_players+1)*(2**(self.num_players-1)) # Player's card & # of bets
        num_bridge_states = self.bridge_len # Bridge positions and waiting state
        self.num_states = [num_kuhn_states + num_bridge_states * num_bridge_states + 1 for _ in range(self.num_players)]
        self.num_actions = [2 + 3 for _ in range(self.num_players)] # 0 = bet (-1), 1 = check (-1), 2 = left, 3 = right, 4 = jump

        self.fict = False

    def start_game(self):
        """Start bridge phase."""
        print(f"game started, fict: {self.fict}")
        import random
        self.game_id = random.randint(0, 1000)
        self.curr_player = 0
        self.rewards = [0 for _ in range(self.num_players)]
        self.ended = False
        self.turn_number = 0

        self.bridge_pos = [0 for _ in range(self.num_players)]
        self.prev_action = [None for _ in range(self.num_players)]

        self.cards = [0,0] # not yet distributed the cards

    def crossed_bridge(self) -> bool:
        """
        Returns:
            bool: True if all players successfully crossed the bridge
        """
        return all([player_pos >= self.bridge_len - 1 for player_pos in self.bridge_pos])

    def start_kuhn(self):
        """Initialize Kuhn poker game."""
        assert self.crossed_bridge(), "All players must have crossed bridge to start"
        assert not self.ended, "Game must not have ended"
        assert all(player_cards == 0 for player_cards in self.cards), "Cards must not have distributed yet"

        # distribute cards
        self.curr_bets = [1 for _ in range(self.num_players)]
        cards_available = [i+1 for i in range(self.num_players+1)]
        random.shuffle(cards_available)
        self.cards = cards_available[:self.num_players]  
        self.folded = [False for _ in range(self.num_players)]
        self.checked = [False for _ in range(self.num_players)]
        self.betted = [False for _ in range(self.num_players)]  

    def finished_round(self) -> bool:
        """If a round (player 1 played, then player two player) has been finished. 
        Necessary because rewards are calculated when a round is finished (and not during mid round).

        Returns:
            bool: True if finished round.
        """
        return self.curr_player == self.num_players - 1
   
    def check_waiting_state(self) -> bool:
        if self.ended:
            return False
        if self.crossed_bridge() and sum(self.cards) == 0:
            # assert not self.finished_round()
            assert all([self.bridge_pos[i] == self.bridge_len for i in range(self.num_players)]), "bridge position must match!"
            return True
        return False

    def observe(self) -> Tuple[int, List, int]:
        """
        Returns observation (2), and the reward:
            Tuple: contains three entries
                - int: Either the card assigned, or bridge position (if in bridge crossing phase), or
                    - (1,2,3) card assigned
                    - (4,5,...) bridge position shifted by 4
                - List[int]: Amount which each player has betted
                - int: Reward, which is zero if episode has not ended
        """

        if not self.ended:
            if self.crossed_bridge():
                if self.check_waiting_state():
                    return self.bridge_pos.copy(), None, 0
                else:
                    return self.cards[self.curr_player], self.curr_bets, 0
            else:
                return self.bridge_pos.copy(), None, 0
        else:
            return -1, None, self.rewards[self.curr_player]

    def action(self, act: str) -> None:
        """
        Execute action, and update state of the enviornment.
        Calculate and update player rewards at the end of each round if:
            - player is in card playing phase & played illegal action
            - player is in bridge crossing phase

        Args:
            act (str): Current player's action to execute
        """
            
        print (f"Game {self.game_id} Player{self.curr_player} executing {act}")
        self.turn_number += 1

        if self.ended:
            print (f"game {self.game_id} executing action on ended game")
            self.curr_player = (self.curr_player + 1) % self.num_players
            return None

        # card playing phase
        elif self.crossed_bridge() and sum(self.cards) > 0:
            if act=="bet":
                self.prev_action[self.curr_player] = -1 # legal action
                self.curr_bets[self.curr_player] += 1
                self.betted[self.curr_player] = True
            elif act=="check":
                self.prev_action[self.curr_player] = -1 # legal action
                if any(self.betted):
                    self.folded[self.curr_player] = True
                else:
                    self.checked[self.curr_player] = True
            elif act=="fold":
                self.prev_action[self.curr_player] = -1 # legal action
                self.folded[self.curr_player] = True
            # choosing left, right, jump will end game
            else:
                self.prev_action[self.curr_player] = 4
            # calculate reward when round finishes due to illegal action
            if 4 in self.prev_action and self.finished_round():
                if self.prev_action[0] != self.prev_action[1]:
                    loser = np.argmax(self.prev_action)
                    self.rewards[loser] -= 10
                    self.rewards[1-loser] += 10
                self.end_game()
            else:
                if all(self.checked):
                    self.end_game()

                elif all(p[0] or p[1] for p in zip(self.betted, self.folded)):
                    self.end_game()
         
        # bridge crossing phase
        else:
            # waiting state - freeze position, but still rewarded
            if self.check_waiting_state():
                self.prev_action[self.curr_player] = self.string_to_act[act]
            # update position
            else:
                curr_position = self.bridge_pos[self.curr_player]
                if act == "left":
                    self.prev_action[self.curr_player] = 2
                    self.bridge_pos[self.curr_player] = max(curr_position - 1, 0)
                elif act == "right":
                    self.prev_action[self.curr_player] = 3
                    self.bridge_pos[self.curr_player] = min(curr_position + 1, self.bridge_len - 1)
                    if self.crossed_bridge() and not self.finished_round():
                        self.bridge_pos[0] = self.bridge_len
                        self.bridge_pos[1] = self.bridge_len

                # jumps/illegal action, so end game
                else:
                    self.prev_action[self.curr_player] = 4

                # calculate reward when round finishes
            if self.finished_round():
                    # same actions, no reward change
                    if self.prev_action[0] == self.prev_action[1]:
                        # both jumps
                        if self.prev_action[0]== 4:
                            self.end_game()

                    # only one plays illegal/jump
                    elif 4 in self.prev_action:
                        loser = np.argmax(self.prev_action)
                        self.rewards[loser] -= 10
                        self.rewards[1-loser] += 10
                        self.end_game()
                    # left right, or right left
                    else:
                        loser = np.argmin(self.prev_action)
                        self.rewards[loser] -= 1
                        self.rewards[1-loser] += 1
                    # We only move on from bridge crossing when round ends because the reward calculation from bridge crossing needs to happen at end of round
                    
                    if (not self.ended) and self.crossed_bridge():
                        self.start_kuhn()

        # if game haven't ended after max turn number passed, we force it to end
        assert sum(self.rewards)==0, "something wrong with game implementation, should be zero sum game"

        if self.turn_number >= self.max_turn_number and (not self.ended) and (not self.crossed_bridge()):
            self.end_game()
        
        print(f"game id: {self.game_id}, fict {self.fict}, prev act: {self.prev_action}, round done: {self.finished_round()}, bridge: {self.bridge_pos}, cards: {self.cards}, ended:{self.ended}, turn# {self.turn_number}")
        
        try:
            _ = self.check_waiting_state()
        except:
            import pdb;pdb.set_trace()

        self.curr_player = (self.curr_player + 1) % self.num_players


    def end_game(self) -> None:
        """Calculate and update player rewards if in card playing phase (added with rewards from bridge crossing)"""
        print(f"Game {self.game_id} ended \n")
        self.ended = True
        if self.crossed_bridge() and 0 not in self.cards:
            if any(self.betted):
                bets = True
            else:
                bets = False
            if bets:
                valid_cards = [self.cards[i] for i in range(self.num_players) if self.betted[i]]
            else:
                valid_cards = self.cards
            best_card = max(valid_cards)

            # if no one selected illegal action (4), they get rewards from card game
            if 4 not in self.prev_action:
                self.winner = self.cards.index(best_card)
                losses = [-bet for bet in self.curr_bets]
                winnings = sum(self.curr_bets)
                card_rewards = losses
                card_rewards[self.winner] += winnings
                self.rewards = list(np.array(card_rewards) + np.array(self.rewards))
     
        print (f"game {self.game_id} rewards: {self.rewards}")
        assert sum(self.rewards) == 0, "something wrong with game implementation, should be zero sum game"

class Bridge_Kuhn_Poker_int_io(Bridge_Kuhn_Poker):
    def __init__(self, _bridge_len=4):
        super().__init__(_bridge_len=_bridge_len)
        self.poss_pots = list(product([1,2],repeat=self.num_players-1))

        all_list = [list(range(self.bridge_len)),list(range(self.bridge_len))]
        self.poss_bridge_pos = list(product(*all_list))
        self.poss_bridge_pos.append((self.bridge_len, self.bridge_len)) # waiting state (4, 4)

        # < max_card_idx: card playing phase, >= max_card_idx: bridge crossing phase
        self.max_card_idx = (len(self.poss_pots)-1) * (self.num_players + 1) + (self.num_players + 1)

    def observe(self) -> Tuple[int]:
        """Returns observation and reward
        - The observation now is encoded into 1-bit: the pot, the card, bridge position

        Returns:
            Tuple[int, int]:
                1) either card (with pot info) or bridge position
                2) reward
                3) turn number
                4) previous actions of the pair
        """
        state, game_pot, reward = super().observe()

        # To encode state into 1 bit
        if state != -1:
            if type(state) == int: # card playing
                card = state
                pot = game_pot.copy()
                pot.pop(self.curr_player)
                try:
                    pot_ind = self.poss_pots.index(tuple(pot))
                except:
                    import pdb;pdb.set_trace()
                return pot_ind * (self.num_players + 1) + card-1, reward, self.turn_number, self.prev_action
            else: # bridge crossing
                curr_bridge_pos = tuple(state)
                curr_bridge_pos_index = self.poss_bridge_pos.index(curr_bridge_pos) + self.max_card_idx
                return curr_bridge_pos_index, reward, self.turn_number, self.prev_action
        else:
            return -1, reward, self.turn_number, self.prev_action

    def action(self, act: int) -> None:
        """Simplifies Kuhn game into two actions

        Args:
            act (int): integer mapping to action string.
            - Somehow call and fold is absorbed.
        """
        try:
            assert self.turn_number <= 12
        except:
            import pdb;pdb.set_trace()
        
        print (f"\nobs: {self.observe()}")
        if act == 0:
            super().action("bet")
        elif act == 1:
            super().action("check")
        elif act == 2:
            super().action("left")
        elif act == 3:
            super().action("right")
        elif act == 4:
            super().action("jump")
        elif act == None:
            super().action("check")
        else:
            raise NotImplementedError("actions must be in range [0,4]")


class Fict_Bridge_Kuhn_int(Bridge_Kuhn_Poker_int_io):

    def __init__(self, _bridge_len=4):
        """self.poss_hidden: List[Tuple[int]] the possible oppponent's card or their position
        [(1,), (2,), (3,), # First 3 is possible opponent's cards (1-3)]
        Then its opponent's possible bridge positions (waiting state included), and last action
        """
        super().__init__(_bridge_len = _bridge_len)

        self.poss_hidden = list(product(list(range(1,self.num_players+2)), repeat=self.num_players-1))
        self.fict = True

    def set_state(self, p_state: int, hidden_state_index: int, p_id: int, turn_number: int, prev_action: list[Union[int, None]]) -> int:
        """Set state using state (observation) and hidden state given.

        Args:
            p_state (int): player state, which is one of either
                - The first max_card_idx is card & pot
                - bridge position
            hidden_state (int): index in poss_hidden, one of either
                - The first 3 index is opponent's card,
                - Then opponent's bridge position and last action
            p_id (int): player id
            turn_number (int)
            prev_action (list[Union[int, None]]))
            
        Returns:
            int: return 0 if legal, -1 illegal
        """
        self.turn_number = turn_number
        self.ended = False

        self.curr_player = p_id
        self.prev_action = prev_action

        # if 3 in self.curr_bets:
        #     import pdb;pdb.set_trace()
            
        if p_state >= self.max_card_idx:
            self.cards = [0 for _ in range(self.num_players)]
            bridge_pos_idx = p_state-self.max_card_idx
            self.bridge_pos = list(self.poss_bridge_pos[bridge_pos_idx]).copy()
            return 0

        else:
            assert hidden_state_index != -1
            self.bridge_pos = [self.bridge_len - 1 for _ in range(self.num_players)]

            self.cards = list(self.poss_hidden[hidden_state_index])
            player_card = (p_state % (self.num_players+1))+1
            self.cards.insert(p_id, player_card)
            if player_card == self.poss_hidden[hidden_state_index]:
                if 3 in self.curr_bets:
                    import pdb;pdb.set_trace()
                return -1 # impossible state
            p_pot = (p_state // (self.num_players+1))

            self.curr_bets = list(self.poss_pots[p_pot])
            self.curr_bets.insert(p_id, 1)
            
            self.betted = [bool(bet-1) for bet in self.curr_bets]
            self.checked = [False for betted in self.betted]
            for p in range(p_id):
                self.checked[p] = not self.betted[p]
            self.folded = [False for player in range(self.num_players)]
            if any(self.betted):
                first = self.betted.index(True)
                if p_id > first:
                    for p in range(first, p_id):
                        self.folded[p] = not self.betted[p]
                else:
                    for p in range(p_id):
                        self.folded[p] = not self.betted[p]
                    for p in range(first, self.num_players):
                        self.folded[p] = not self.betted[p]
            
            return 0

    def get_hidden(self, p_id: int) -> int:
        """
        Args:
            p_id (int): state

        Returns:
            int: index in poss_hidden, one of either
            - The first 3 index is opponent's card,
            - Then opponent's bridge position & prev action
        """
        if sum(self.cards) == 0:
            return -1
        else:
            curr_cards = self.cards.copy()
            curr_cards.pop(p_id) # the remaining left is opponent's card
            assert curr_cards[0]!=0, f"card cannot be 0, player: {p_id}, bridge_pos: {self.bridge_pos}, cards:{self.cards}, crossed_bridge:{self.crossed_bridge()}"
            hidden_state_index = self.poss_hidden.index(tuple(curr_cards))

        assert hidden_state_index < len(self.poss_hidden), f"hidden state {hidden_state_index} is out of bound"
        return hidden_state_index
