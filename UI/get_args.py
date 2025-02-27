from games.bridge_kuhn import Bridge_Kuhn_Poker_int_io, Fict_Bridge_Kuhn_int
from games.kuhn import Kuhn_Poker_int_io, Fict_Kuhn_int
from games.leduc import leduc_int, leduc_fict
import agents.learners as learners
import sys
import time
import logging
log = logging.getLogger(__name__)


def run():
    bridge_len = 0
    if len(sys.argv) > 1:
        if '--lvls' in sys.argv:
            level_ind = sys.argv.index('--lvls')
            if len(sys.argv) > level_ind:
                try:
                    num_lvls = int(sys.argv[level_ind+1])
                except TypeError:
                    print("Please enter a numerical value for number of levels")
                    return -1
            else:
                print("Please enter number of levels")
                return(-1)
        else:
            num_lvls = 10
        if '--game' in sys.argv:
            game_ind = sys.argv.index('--game')
            if len(sys.argv) > game_ind:
                game_name = sys.argv[game_ind+1]
                if game_name  == "kuhn":
                    game = Kuhn_Poker_int_io()
                    fict_game = Fict_Kuhn_int()
                    exploit_learner = learners.kuhn_exact_solver()
                elif game_name == "bridge_kuhn":
                    bridge_len = 4
                    if '--bridge_len' in sys.argv:
                        bridge_len_idx = sys.argv.index('--bridge_len')
                        bridge_len = int(sys.argv[bridge_len_idx+1])
                    game = Bridge_Kuhn_Poker_int_io(_bridge_len=bridge_len)
                    fict_game = Fict_Bridge_Kuhn_int(_bridge_len=bridge_len)
                    exploit_learner = learners.bridge_kuhn_exact_solver(bridge_len)
                elif game_name == "leduc":
                    game = leduc_int()
                    fict_game = leduc_fict()
                    exploit_learner = learners.actor_critic(learners.softmax, learners.value_advantage, \
                                            game.num_actions[0], game.num_states[0], tol=9999)
                else:
                    print("Please enter a game choice")
                    return -1
            else:
                print("Please select a game")
                return(-1)
        else:
            game_name = "kuhn"
            game = Kuhn_Poker_int_io()
            fict_game = Fict_Kuhn_int()
            exploit_learner = learners.kuhn_exact_solver()
            
    else:
        num_lvls = 10
        game = Kuhn_Poker_int_io()
        fict_game = Fict_Kuhn_int()
        exploit_learner = learners.kuhn_exact_solver()
    if '--all_avg' in sys.argv or '-a' in sys.argv:
        averaged_bel = True
        averaged_pol = True
        learn_with_avg = True
    else:
        averaged_bel ='--avg_bel' in sys.argv or '-ab' in sys.argv
        averaged_pol ='--avg_pol' in sys.argv or '-ap' in sys.argv
        learn_with_avg = '--avg_learn' in sys.argv or '-al' in sys.argv
    if '--debug' in sys.argv:
        logging.basicConfig(level=logging.DEBUG,\
                format='%(relativeCreated)6d %(threadName)s %(message)s')
    elif '-v' in sys.argv or '--verbose' in sys.argv:
        logging.basicConfig(level=logging.INFO,\
                format='%(relativeCreated)6d %(threadName)s %(message)s')
    if '--learner' in sys.argv:
        learn_ind = sys.argv.index('--learner')
        if len(sys.argv) > learn_ind:
            learner_type = sys.argv[learn_ind+1]
        else:
            print("Please select a learner")
            return(-1)
    else:
        learner_type='obl'
    if '--fsp' in sys.argv:
        mode = 'fsp'
    else:
        mode = 'obl'
    opts = {"num_lvls":num_lvls,
            "game_name":game_name,
            "game":game,
            "fict_game":fict_game,
            "exploit_learner":exploit_learner,
            "avg_bel":averaged_bel,
            "avg_pol":averaged_pol,
            "learn_w_avg":learn_with_avg,
            "learner_type":learner_type,
            "mode":mode,
            "bridge_len": bridge_len
            }
    return opts 
