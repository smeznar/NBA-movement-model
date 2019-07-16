import csv
import json
import math
import random
import time

import numpy as np
from PIL import Image
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
        if len(self.data[Constants.MD_EVENT_STR]) > 0:
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
            #self.right_direction = self.get_starting_direction()

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
            for moment in event[Constants.MD_MOMENTS_STR]:
                new_period = False
                if moment[Constants.MD_TIME_LEFT_NUM] > period_time and moment[Constants.MD_PERIOD_NUM] == period:
                    continue
                else:
                    p = period
                    period = moment[Constants.MD_PERIOD_NUM]
                    new_period = p != period
                    period_time = moment[Constants.MD_TIME_LEFT_NUM]
                if moment[Constants.MD_SHOTCLOCK_NUM] is None:
                    if len(attack) > 0:
                        self.all_attacks.append(attack)
                        self.moment_numbers.append(i)
                    attack = list()
                    time = 25.0
                    continue
                elif moment[Constants.MD_SHOTCLOCK_NUM] > time or new_period:
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
            #if moment[Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM] > ball_x:
            #    positive += 1
            #elif moment[Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM] < ball_x:
            #    negative += 1
            x = moment[Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM]
            y = moment[Constants.MD_POSITONS_NUM][0][Constants.MD_Y_COORD_NUM]
            min_distance = 13600
            team_num = -1
            for i in range(1, len(moment[Constants.MD_POSITONS_NUM])):
                distance = math.pow(x - moment[Constants.MD_POSITONS_NUM][i][Constants.MD_X_COORD_NUM], 2) \
                           + math.pow(y - moment[Constants.MD_POSITONS_NUM][i][Constants.MD_Y_COORD_NUM], 2)
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
        attacks = list()
        while self.event_num < len(self.all_attacks):
            attack = self.all_attacks[self.event_num]
            moment_number = self.moment_numbers[self.event_num]
            self.event_num += 1
            if self.is_attack_wanted(attack, moment_number):
                a = self.transform_attack_to_vector(attack, moment_number)
                b = self.attack_to_picture(attack, moment_number)
                if a is not None:
                    attacks.append((a, b))
        ret = self.change_positions(attacks)
        if len(ret) > 0:
            attacks, images = ret
            i = 0
            for im in images:
                image = Image.fromarray(im, 'L')
                filename = self.team_name + "_" + self.opponent_name + "_" + str(i) + "_" + str(time.time()) + ".jpg"
                image.save('../match_data/images/' + filename)
                i += 1
            return attacks
        else:
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
        if (start_time - end_time) < 5 or (start_time - end_time) > 24:
            return None
        # modified_attack = self.remove_static_moments(attack)
        num_of_fields = 10
        modified_attack = attack
        if len(modified_attack) < num_of_fields:
            modified_attack = attack
        vector = list()
        step = len(modified_attack)/(num_of_fields+1)
        next_moment = step
        last_x = modified_attack[0][Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM]
        last_y = modified_attack[0][Constants.MD_POSITONS_NUM][0][Constants.MD_Y_COORD_NUM]
        i = 0
        while i < num_of_fields:
            moment = modified_attack[min(int(next_moment), len(attack)-1)]
            x = moment[Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM]
            y = moment[Constants.MD_POSITONS_NUM][0][Constants.MD_Y_COORD_NUM]
            vector.append(last_x)
            vector.append(last_y)
            vector.append(x-last_x)
            vector.append(y-last_y)
            vector = self.add_opponent_players(moment, vector)
            last_x = x
            last_y = y
            next_moment += step
            i += 1
        vector.append(start_time-end_time)
        # vector.append(math.sqrt(math.pow(88-last_x, 2)+math.pow(25-last_y, 2)))
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

    def add_opponent_players(self, moment, vector):
        def sort_val(i):
            return i[2]
        positions = list()
        x = moment[Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM]
        y = moment[Constants.MD_POSITONS_NUM][0][Constants.MD_Y_COORD_NUM]

        for player in moment[Constants.MD_POSITONS_NUM]:
            if player[0] not in [-1, self.team_number]:
                x_player = player[Constants.MD_X_COORD_NUM]
                y_player = player[Constants.MD_Y_COORD_NUM]
                distance = math.sqrt(math.pow(x_player-x, 2)+math.pow(y_player-y, 2))
                positions.append([x_player, y_player, distance])

        positions.sort(key=sort_val)
        for v in positions:
            vector.append(v[0])
            vector.append(v[1])
        return vector

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
            norms = (norm(array([last_delta_x, last_delta_y])) * norm(array([delta_x, delta_y])))
            if norms != 0:
                c = dot(array([last_delta_x, last_delta_y]), array([delta_x, delta_y]))\
                    / norms
            else:
                c = 0
            angle = arccos(clip(c, -1, 1))
            if abs(min(angle, 6.28-angle)) > 0.5:
                num_of_passes += 1
            last_x = x
            last_y = y
            last_delta_x = delta_x
            last_delta_y = delta_y
        return num_of_passes

    def change_positions(self, attacks):
        if len(attacks) < 1:
            return attacks
        last_period = 1
        buffered_attacks = list()
        buffered_image = list()
        corrected_attacks = list()
        corrected_image = list()
        i = 0
        right_side = 0
        wrong_side = 0
        while i < len(attacks):
            attack = attacks[i][0]
            if len(attack) != 146:
                i += 1
                continue
            if attack[142] != last_period:
                last_period = attack[142]
                if right_side < wrong_side:
                    buffered_attacks = self.transform_row_coordinates(buffered_attacks)
                    for im in buffered_image:
                        corrected_image.append(np.flip(im, axis=(0, 1)))
                else:
                    for im in buffered_image:
                        corrected_image.append(im)
                corrected_attacks = corrected_attacks + buffered_attacks
                buffered_attacks = list()
                buffered_image = list()
                right_side = 0
                wrong_side = 0
            if attack[126] > 47:
                right_side += 1
            else:
                wrong_side += 1
            if len(attack) == 146:
                buffered_attacks.append(attack)
            if attacks[i][1] is not None:
                buffered_image.append(attacks[i][1])
            i += 1
        if right_side < wrong_side:
            buffered_attacks = self.transform_row_coordinates(buffered_attacks)
            for im in buffered_image:
                corrected_image.append(np.flip(im, axis=(0, 1)))
        else:
            for im in buffered_image:
                corrected_image.append(im)
        corrected_attacks = corrected_attacks + buffered_attacks
        for a in corrected_attacks:
            a.insert(141, math.sqrt(math.pow(88 - a[126], 2) + math.pow(25 - a[127], 2)))
        return corrected_attacks, corrected_image

    def transform_row_coordinates(self, buffered_attacks):
        for a in buffered_attacks:
            for i in range(10):
                a[i * 14] = Constants.COURT_DIMENSION_X - a[i * 14]
                a[i * 14 + 1] = Constants.COURT_DIMENSION_Y - a[i * 14 + 1]
                a[i * 14 + 2] = - a[i * 14 + 2]
                a[i * 14 + 3] = - a[i * 14 + 3]
                for j in range(5):
                    a[i * 14 + j * 2 + 4] = Constants.COURT_DIMENSION_X - a[i * 14 + j * 2 + 4]
                    a[i * 14 + j * 2 + 5] = Constants.COURT_DIMENSION_Y - a[i * 14 + j * 2 + 5]
        return buffered_attacks

    def attack_to_picture(self, attack, moment_number):
        start_time = attack[0][Constants.MD_TIME_LEFT_NUM]
        end_time = attack[-1][Constants.MD_TIME_LEFT_NUM]
        if end_time > 120 or (start_time - end_time) < 5 or (start_time - end_time) > 24:
            return None
        data = np.zeros((94, 50), dtype=np.uint8)
        step = 200/len(attack)
        point = 55
        #last_x = attack[0][Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM]
        #last_y = attack[0][Constants.MD_POSITONS_NUM][0][Constants.MD_Y_COORD_NUM]
        for a in attack:
            x = a[Constants.MD_POSITONS_NUM][0][Constants.MD_X_COORD_NUM]
            y = a[Constants.MD_POSITONS_NUM][0][Constants.MD_Y_COORD_NUM]
            # Draw line to array
            if 0 <= x < 94 and 0 <= y < 50:
                data[int(x), int(y)] = int(point)
            #
            #last_x = x
            #last_y = y
            point += step
        #plt.imshow(data, cmap="gray")
        #plt.show()
        #a = 0
        #data = np.flip(data, axis=(0, 1))
        return data


# for outliers:
# https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame?fbclid=IwAR0F0_gPsovw9jErz3kAjHjip-whp8Q4hm2cZBHtrxNpHLnIMuyFdvO8TGc

def transform_vector_for_nn(in_name, out_name):
    rows = list()
    with open(in_name) as file:
        reader = csv.reader(file, delimiter='|')
        for row in reader:
            if len(row) != 147:
                continue
            #if float(row[141])>50:
            #    continue
            rows.append(row)
    random.shuffle(rows)
    with open(out_name, 'w') as file:
        writer = csv.writer(file, delimiter='|')
        for r in rows:
            writer.writerow(r)


if __name__ == '__main__':
    transform_vector_for_nn('../match_data/nba_vectors_defenders_distance.csv',
                            '../match_data/nba_vectors_defenders_distance_corrected.csv')
