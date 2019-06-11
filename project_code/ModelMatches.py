import glob
import os

import numpy as np
from pyunpack import Archive

from project_code import MatchVectorizer


class FileExtractor:
    def __init__(self):
        self.base_path = '/Volumes/Seagate Expansion Drive/nba-movement-data/data/'
        self.teams = ['OKC', 'GSW', 'MEM']
        self.current_team = ''
        # OKC fast not many passes, GSW fast and many passes, MEM slow and many passes
        self.match_list = []
        self.team_count = 0
        self.match_count = 0

    def get_next_file_name(self):
        for file in glob.glob('../match_data/match_event/*'):
            os.remove(file)
        if self.match_count == len(self.match_list):
            if self.get_next_teams_matches():
                return None
        match_path = self.match_list[self.match_count]
        m_name = match_path.split('/')[-1].split('.')
        print('Modeling next match: ' + m_name[3] + ' vs ' + m_name[5])
        Archive(match_path).extractall('../match_data/match_event/')
        match_name = glob.glob('../match_data/match_event/*')
        self.match_count += 1
        return match_name[0]

    def get_next_teams_matches(self):
        self.match_count = 0
        if self.team_count >= len(self.teams):
            return True
        self.current_team = self.teams[self.team_count]
        self.match_list = glob.glob(self.base_path + '*' + self.teams[self.team_count] + '*.7z')
        self.team_count += 1
        return False


class ModelMatches:
    def __init__(self):
        self.x = list()
        self.y = list()

    def add_row(self, x, y):
        self.x.append(x)
        self.y.append(y)

    def create_a_model(self):
        x = np.array(self.x)
        y = np.array(self.y)
        pass


# TODO: maybe transfer directly to ModelMatches
if __name__ == '__main__':
    fe = FileExtractor()
    mm = ModelMatches()
    file_name = fe.get_next_file_name()
    while file_name is not None:
        mv = MatchVectorizer.MatchVectorizer(file_name, fe.current_team)
        print(fe.current_team, ': ', file_name)
        attack = mv.get_next_attack()
        i = 0
        while attack is not None:
            i += 1
            mm.add_row(attack, fe.current_team)
            attack = mv.get_next_attack()
        print(i)
        del mv
        file_name = fe.get_next_file_name()
    mm.create_a_model()

    print('Finished')
