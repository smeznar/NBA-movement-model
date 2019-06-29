import json
import math

from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm

from project_code import Constants


class MatchVectorizer:
    def __init__(self, file_path, team_name):
        self.WANTED_ATTACK_TIME = 120
        self.PERIODS_WANTED = [1, 2, 3, 4]
        self.data = self.read_file(file_path)
        self.all_attacks = list()
        self.moment_numbers = list()
        self.divide_attacks()
        self.num_of_vector_places = 4
        self.event_num = 0
        self.team_number = 0
        self.team_name = team_name
        self.opponent_name = ''
        if self.data[Constants.MD_EVENT_STR][0][Constants.MD_OPPONENT_TEAM_STR][
            Constants.MD_ABBREVIATION_STR] == team_name:
            self.team_number = self.data[Constants.MD_EVENT_STR][0][Constants.MD_OPPONENT_TEAM_STR][
                Constants.MD_TEAMID_STR]
            self.opponent_name = self.data[Constants.MD_EVENT_STR][0][Constants.MD_HOME_TEAM_STR][
                Constants.MD_ABBREVIATION_STR]
        else:
            self.team_number = self.data[Constants.MD_EVENT_STR][0][Constants.MD_HOME_TEAM_STR][Constants.MD_TEAMID_STR]
            self.opponent_name = self.data[Constants.MD_EVENT_STR][0][Constants.MD_OPPONENT_TEAM_STR][
                Constants.MD_ABBREVIATION_STR]
        self.right_direction = self.get_starting_direction()

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

    def divide_attacks(self):
        time = 25.0
        period_time = 800
        period = 0
        attack = list()
        num = 0
        for i in range(len(self.data[Constants.MD_EVENT_STR])):
            num = i
            event = self.data[Constants.MD_EVENT_STR][i]
            # time = event[Constants.MD_MOMENTS_STR][0][Constants.MD_SHOTCLOCK_NUM]
            for moment in event[Constants.MD_MOMENTS_STR]:
                if moment[Constants.MD_PERIOD_NUM] == period and moment[Constants.MD_TIME_LEFT_NUM] > period_time:
                    continue
                else:
                    period = moment[Constants.MD_PERIOD_NUM]
                    period_time = moment[Constants.MD_TIME_LEFT_NUM]
                if moment[Constants.MD_SHOTCLOCK_NUM] is None:
                    continue
                elif moment[Constants.MD_SHOTCLOCK_NUM] > time:
                    if len(attack) > 0:
                        self.all_attacks.append(attack)
                        self.moment_numbers.append(i)
                    attack = list()
                    attack.append(moment)
                else:
                    attack.append(moment)
                time = moment[Constants.MD_SHOTCLOCK_NUM]
        if len(attack) > 0:
            self.all_attacks.append(attack)
            self.moment_numbers.append(num)

    def get_starting_direction(self):
        if len(self.all_attacks) < 1:
            return
        attack = self.all_attacks[0]
        positive = 0
        negative = 0
        our_team = 0
        opponents = 0
        ball_x = attack[0][Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM]
        # TODO: Maybe only first and last moment?
        for moment in attack:
            if moment[Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM] > ball_x:
                positive += 1
            elif moment[Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM] < ball_x:
                negative += 1
            ball_x = moment[Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM]
            ball_y = moment[Constants.MD_POSITONS_NUM][0][Constants.MD_Y_COORD_NUM]
            min_distance = 13600
            team_num = -1
            for i in range(1, len(moment[Constants.MD_POSITONS_NUM])):
                distance = math.pow(ball_x - moment[Constants.MD_POSITONS_NUM][i][Constants.MD_X_COORD_NUM], 2) \
                           + math.pow(ball_y - moment[Constants.MD_POSITONS_NUM][i][Constants.MD_Y_COORD_NUM], 2)
                if distance < min_distance:
                    min_distance = distance
                    team_num = moment[Constants.MD_POSITONS_NUM][i][Constants.MD_PLAYERS_TEAMID_NUM]
            if team_num == self.team_number:
                our_team += 1
            else:
                opponents += 1
        result = 0
        if our_team < opponents:
            result += 1
        if positive > negative:
            result += 1
        return result % 2

    def get_next_attack(self):
        while self.event_num < len(self.all_attacks):
            attack = self.all_attacks[self.event_num]
            moment_number = self.moment_numbers[self.event_num]
            self.event_num += 1
            if self.is_attack_wanted(attack, moment_number):
                a = self.transform_attack_to_vector(attack, moment_number)
                if a is not None:
                    return a
        return None

    def is_attack_wanted(self, attack, moment_number):
        # if not self.is_time_ok(attack):
        #    return False
        if not self.is_the_right_team_attacking(attack):
            return False
        return True

    # Returns False if attack ends more than "time" in the period
    def is_time_ok(self, attack):
        if attack[-1][Constants.MD_TIME_LEFT_NUM] > self.WANTED_ATTACK_TIME \
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
                distance = math.pow(ball_x - a[Constants.MD_POSITONS_NUM][i][Constants.MD_X_COORD_NUM], 2) \
                           + math.pow(ball_y - a[Constants.MD_POSITONS_NUM][i][Constants.MD_Y_COORD_NUM], 2)
                if distance < min_distance:
                    min_distance = distance
                    team_num = a[Constants.MD_POSITONS_NUM][i][Constants.MD_PLAYERS_TEAMID_NUM]
            if team_num == self.team_number:
                our_team += 1
            else:
                opponents += 1
        return our_team > opponents

    def transform_attack_to_vector(self, attack, moment_number):
        start_time = attack[0][Constants.MD_TIME_LEFT_NUM]
        end_time = attack[-1][Constants.MD_TIME_LEFT_NUM]
        if start_time - end_time < 5:
            return None
        modified_attack = self.remove_static_moments(attack)
        num_of_fields = 10
        vector = list()
        step = len(modified_attack)/(num_of_fields+1)
        next = step
        last_x = 0
        last_y = 0
        if self.right_direction + modified_attack[0][Constants.MD_PERIOD_NUM] % 2 == 1:
            last_x = Constants.COURT_DIMENSION_X - modified_attack[0][Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM]
            last_y = Constants.COURT_DIMENSION_Y - modified_attack[0][Constants.MD_POSITONS_NUM][0][Constants.MD_Y_COORD_NUM]
        else:
            last_x = modified_attack[0][Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM]
            last_y = modified_attack[0][Constants.MD_POSITONS_NUM][0][Constants.MD_Y_COORD_NUM]
        i = 0
        while i < num_of_fields:
            moment = modified_attack[min(int(next), len(attack)-1)]
            x = 0
            y = 0
            if self.right_direction + moment[Constants.MD_PERIOD_NUM] % 2 == 1:
                x = Constants.COURT_DIMENSION_X - moment[Constants.MD_POSITONS_NUM][0][
                    Constants.MD_X_COORD_NUM]
                y = Constants.COURT_DIMENSION_Y - moment[Constants.MD_POSITONS_NUM][0][
                    Constants.MD_Y_COORD_NUM]
            else:
                x = moment[Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM]
                y = moment[Constants.MD_POSITONS_NUM][0][Constants.MD_Y_COORD_NUM]
            vector.append(last_x)
            vector.append(last_y)
            vector.append(x-last_x)
            vector.append(y-last_y)
            last_x = x
            last_y = y
            next += step
            i += 1
        vector.append(start_time-end_time)
        vector.append(math.sqrt(math.pow(88-last_x, 2)+math.pow(25-last_y, 2)))
        vector.append(self.get_num_of_passes(modified_attack))
        vector.append(attack[-1][Constants.MD_PERIOD_NUM])
        vector.append(attack[-1][Constants.MD_TIME_LEFT_NUM])
        vector.append(self.opponent_name)
        vector.append(self.team_name)
        return vector

    def remove_static_moments(self, attack):
        ma = list()
        last_x = 0
        last_y = 0
        for m in attack:
            if (abs(m[Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM] - last_x)
                + abs(m[Constants.MD_POSITONS_NUM][0][Constants.MD_Y_COORD_NUM] - last_y)) > 5:
                last_x = m[Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM]
                last_y = m[Constants.MD_POSITONS_NUM][0][Constants.MD_Y_COORD_NUM]
                ma.append(m)
        ma.append(attack[-1])
        return ma

    def get_num_of_passes(self, attack):
        if len(attack) < 3:
            return 0
        num_of_passes = 0
        last_x = attack[1][Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM]
        last_y = attack[1][Constants.MD_POSITONS_NUM][0][Constants.MD_Y_COORD_NUM]
        last_delta_x = last_x - attack[0][Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM]
        last_delta_y = last_y - attack[0][Constants.MD_POSITONS_NUM][0][Constants.MD_Y_COORD_NUM]
        for i in range(2, len(attack)):
            x = attack[i][Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM]
            y = attack[i][Constants.MD_POSITONS_NUM][0][Constants.MD_Y_COORD_NUM]
            delta_x = x - last_x
            delta_y = y - last_y
            c = dot(array([last_delta_x, last_delta_y]), array([delta_x, delta_y]))\
                / norm(array([last_delta_x, last_delta_y])) / norm(array([delta_x, delta_y]))  # -> cosine of the angle
            angle = arccos(clip(c, -1, 1))
            if abs(min(angle, 6.28-angle)) > 0.5:
                num_of_passes += 1
            last_x = x
            last_y = y
            last_delta_x = delta_x
            last_delta_y = delta_y
        return num_of_passes
