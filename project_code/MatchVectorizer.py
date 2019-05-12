import json


class MatchVectorizer:
    def __init__(self, file_path):
        self.data = self.read_file(file_path)

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

    def get_next_attack(self):
        return None

    def is_attack_wanted(self):
        pass
