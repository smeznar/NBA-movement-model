import csv
import glob
import os

from pyunpack import Archive

from project_code import Constants
from project_code import MatchVectorizer

all_teams = ['BOS', 'LAC', 'CHI', 'PHI', 'CLE', 'MEM', 'ATL', 'SAC', 'UTA', 'POR', 'MIN', 'OKC', 'TOR', 'BKN', 'NOP',
             'GSW', 'MIA', 'DAL', 'PHX', 'IND', 'SAS', 'MIL', 'DET', 'NYK', 'CHA', 'WAS', 'HOU', 'LAL', 'DEN', 'ORL']


class FileExtractor:
    def __init__(self):
        self.base_path = Constants.DATA_BASE_PATH
        self.teams = ['ORL']
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
        if len(match_name) > 0:
            return match_name[0]
        else:
            return 0

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
        self.file = open('nba_vectors_defenders_distance_all_moments.csv', 'a')
        self.writer = csv.writer(self.file, delimiter='|')

    def write_row(self, x):
        self.writer.writerow(x)

    def finish_file(self):
        self.file.close()


# TODO: maybe transfer directly to ModelMatches
if __name__ == '__main__':
    fe = FileExtractor()
    #mm = ModelMatches()
    file_name = fe.get_next_file_name()
    while file_name is not None:
        mv = MatchVectorizer.MatchVectorizer(file_name, fe.current_team)
        print(fe.current_team, ': ', file_name)
        attacks = mv.get_next_attack()
#        print(len(attacks))
        #
        #for a in attacks:
        #    mm.write_row(a)
        # except Exception as e:
        #     mm.finish_file()
        #     print(e)
        del mv
        file_name = fe.get_next_file_name()
        while file_name == 0:
            file_name = fe.get_next_file_name()
    #mm.finish_file()
