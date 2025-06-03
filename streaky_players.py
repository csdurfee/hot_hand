import numpy as np

rng = np.random.default_rng(2718)

class BasePlayer:
    def __init__(self, shooting_percentage):
        self.rng = rng
        self.shooting_percentage = shooting_percentage        
        self.game_history = []
        self.shoot_pct_history = []

    def get_shooting_percentage(self):
        ## base behavior -- always return the same shooting percentage
        return self.shooting_percentage
    
    def get_rng(self):
        return self.rng.random()
    
    def take_shot(self):
        shoot_pct = self.get_shooting_percentage()
        # this allows me to track the shooting percentage changes due to streakiness
        self.shoot_pct_history.append(shoot_pct)
        if self.get_rng() < shoot_pct:
            make_or_miss = 1
        else:
            make_or_miss = 0
        self.game_history.append(make_or_miss)
        return make_or_miss
    
    def end_game(self):
        self.game_history = []
        self.shoot_pct_history = []

class LukewarmPlayer(BasePlayer):
    # when player has made more than lower/upper thresh, their shooting percentage 
    # gets boosted/penalized.    
    def __init__(self, shooting_percentage):
        self.BOOST_POS_AMOUNT = .2
        self.BOOST_NEG_AMOUNT = -.2
        self.LOWER_THRESH = .2
        self.UPPER_THRESH = .8
        self.MIN_ATTEMPTS = 4

        return super().__init__(shooting_percentage)

    def get_shooting_percentage(self):
        attempts = len(self.game_history)
        makes    = sum(self.game_history) 
            
        if attempts < self.MIN_ATTEMPTS:
            return self.shooting_percentage
        else:
            # streaky behavior is in play
            game_percentage = makes / attempts
            if game_percentage < self.LOWER_THRESH:
                # boost shooting percentage when game percentage is low
                return min(self.shooting_percentage + self.BOOST_POS_AMOUNT, 1)
            elif game_percentage > self.UPPER_THRESH:
                # penalize shooting percentage when game percentage is high
                return max(self.shooting_percentage + self.BOOST_NEG_AMOUNT, 0)
            else:
                ## player is currently not on a hot/cold streak
                return self.shooting_percentage

class OnlyHeatCheckPlayer(LukewarmPlayer):
    """
    like lukewarm player but they only get a penalty to fg%age when they are shooting well.
    """
def __init__(self, shooting_percentage):
        super().__init__(shooting_percentage)
        self.LOWER_THRESH = 0
        self.BOOST_NEG_AMOUNT = -.3

class GetABucketPlayer(LukewarmPlayer):
    """
    like lukewarm player but they only get a boost to fg %age when they are shooting poorly,
    """
    def __init__(self, shooting_percentage):
        super().__init__(shooting_percentage)
        self.UPPER_THRESH = 1
        self.BOOST_POS_AMOUNT = .3


class TrulyStreakyPlayer(LukewarmPlayer):
    """
    the mythical unicorn, they shoot better after they've made a few in a row
    and shoot worse after they've missed a few in a row.

    note: the window for "in a row" is still the whole game

    """
    def __init__(self, shooting_percentage):
        super().__init__(shooting_percentage)
        self.BOOST_POS_AMOUNT = -.2 # subtract 20% from shooting percentage when "ice cold"
        self.BOOST_NEG_AMOUNT =  .2 # add 20% to shooting percentage when "heating up"

        # I messed around with the thresholds but ended up back with the defaults for the 
        # LukewarmPlayer.
        self.UPPER_THRESH = .8
        self.LOWER_THRESH = .2
        self.MIN_ATTEMPTS = 4