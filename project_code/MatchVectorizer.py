import json
import math

class MatchVectorizer:
    def __init__(self, file_path, team_name):
        self.data = self.read_file(file_path)
        self.all_attacks = list()
        self.moment_numbers = list()
        self.divide_attacks()
        self.event_num = 0
        self.team_number = 0
        if self.data['events'][0]['visitor']['abbreviation'] == team_name:
            self.team_number = self.data['events'][0]['visitor']['teamid']
        else:
            self.team_number = self.data['events'][0]['home']['teamid']

    @staticmethod
    def read_file(file_path):
        with open(file_path) as f:
            data = json.load(f)
            no_moment = []
            for i in range(len(data['events'])):
                if len(data['events'][i]['moments']) < 1:
                    no_moment.append(i)
            for i in reversed(no_moment):
                del data['events'][i]
            return data

    def divide_attacks(self):
        for i in range(len(self.data['events'])):
            event = self.data['events'][i]
            time = event['moments'][0][3]
            attack = list()
            for moment in event['moments']:
                if time is None:
                    if moment[3] is None:
                        continue
                    else:
                        time = moment[3]
                if moment[3] is None:
                    continue
                elif moment[3] > time:
                    if len(attack) > 0:
                        self.all_attacks.append(attack)
                        self.moment_numbers.append(i)
                    time = moment[3]
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
        if not self.is_time_ok(attack, 120, [1, 2, 3, 4]):
            return False
        if not self.is_the_right_team_attacking(attack):
            return False
        return True

    # Returns False if attack ends more than "time" in the period
    def is_time_ok(self, attack, time, period):
        if attack[-1][2] > time and attack[0][0] in period:
            return False
        else:
            return True

    # Returns True if the team we want is closer to the ball more times during the attack
    def is_the_right_team_attacking(self, attack):
        our_team = 0
        opponents = 0
        for a in attack:
            # Todo: safe checks maybe?
            ball_x = a[5][0][2]
            ball_y = a[5][0][3]
            min_distance = 100000000000000
            team_num = -1
            for i in range(1, len(a[5])):
                distance = math.pow(ball_x - a[5][i][2], 2) + math.pow(ball_y - a[5][i][2], 2)
                if distance < min_distance:
                    min_distance = distance
                    team_num = a[5][i][0]
            if team_num == self.team_number:
                our_team += 1
            else:
                opponents += 1
        return our_team > opponents
