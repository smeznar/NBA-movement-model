import json
import math

from project_code import Constants


class MatchVectorizer:
    def __init__(self, file_path, team_name):
        self.WANTED_ATTACK_TIME = 120
        self.PERIODS_WANTED = [1, 2, 3, 4]
        self.data = self.read_file(file_path)
        self.all_attacks = list()
        self.moment_numbers = list()
        self.divide_attacks()
        self.event_num = 0
        self.team_number = 0
        if self.data[Constants.MD_EVENT_STR][0][Constants.MD_OPPONENT_TEAM_STR][Constants.MD_ABBREVIATION_STR] == team_name:
            self.team_number = self.data[Constants.MD_EVENT_STR][0][Constants.MD_OPPONENT_TEAM_STR][
                Constants.MD_TEAMID_STR]
        else:
            self.team_number = self.data[Constants.MD_EVENT_STR][0][Constants.MD_HOME_TEAM_STR][Constants.MD_TEAMID_STR]

    @staticmethod
    def read_file(file_path):
        with open(file_path) as f:
            data = json.load(f)
            no_moment = []
            for i in range(len(data[Constants.MD_EVENT_STR])):
                if len(data[Constants.MD_EVENT_STR][i][Constants.MD_MOMENTS_STR]) < 1:
                    no_moment.append(i)
            for i in reversed(no_moment):
                del data[Constants.MD_EVENT_STR][i]
            return data

    # TODO: probably there are attacks that span more than one event
    def divide_attacks(self):
        time = 25.0
        for i in range(len(self.data[Constants.MD_EVENT_STR])):
            event = self.data[Constants.MD_EVENT_STR][i]
            time = event[Constants.MD_MOMENTS_STR][0][Constants.MD_SHOTCLOCK_NUM]
            attack = list()
            for moment in event[Constants.MD_MOMENTS_STR]:
                if time is None:
                    if moment[Constants.MD_SHOTCLOCK_NUM] is None:
                        continue
                    else:
                        time = moment[Constants.MD_SHOTCLOCK_NUM]
                if moment[Constants.MD_SHOTCLOCK_NUM] is None:
                    continue
                elif moment[Constants.MD_SHOTCLOCK_NUM] > time:
                    if len(attack) > 0:
                        self.all_attacks.append(attack)
                        self.moment_numbers.append(i)
                    time = moment[Constants.MD_SHOTCLOCK_NUM]
                    attack = [moment]
                else:
                    attack.append(moment)
            if len(attack) > 0:
                self.all_attacks.append(attack)
                self.moment_numbers.append(i)

    def get_next_attack(self):
        while self.event_num < len(self.all_attacks):
            attack = self.all_attacks[self.event_num]
            moment_number = self.moment_numbers[self.event_num]
            if self.is_attack_wanted(attack, moment_number):
                # Do something
                self.event_num += 1
                return True
            else:
                self.event_num += 1
        return None

    def is_attack_wanted(self, attack, moment_number):
        if not self.is_time_ok(attack):
            return False
        if not self.is_the_right_team_attacking(attack):
            return False
        return True

    # Returns False if attack ends more than "time" in the period
    def is_time_ok(self, attack):
        if attack[-1][Constants.MD_TIME_LEFT_NUM] > self.WANTED_ATTACK_TIME\
                and attack[0][Constants.MD_PERIOD_NUM] in self.PERIODS_WANTED:
            return False
        else:
            return True

    # Returns True if the team we want is closer to the ball more times during the attack
    def is_the_right_team_attacking(self, attack):
        our_team = 0
        opponents = 0
        for a in attack:
            # Todo: safe checks maybe?
            ball_x = a[Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM]
            ball_y = a[Constants.MD_POSITONS_NUM][0][Constants.MD_Y_COORD_NUM]
            min_distance = 13600
            team_num = -1
            for i in range(1, len(a[Constants.MD_POSITONS_NUM])):
                distance = math.pow(ball_x - a[Constants.MD_POSITONS_NUM][i][Constants.MD_X_COORD_NUM], 2)\
                           + math.pow(ball_y - a[Constants.MD_POSITONS_NUM][i][Constants.MD_Y_COORD_NUM], 2)
                if distance < min_distance:
                    min_distance = distance
                    team_num = a[Constants.MD_POSITONS_NUM][i][Constants.MD_PLAYERS_TEAMID_NUM]
            if team_num == self.team_number:
                our_team += 1
            else:
                opponents += 1
        return our_team > opponents

