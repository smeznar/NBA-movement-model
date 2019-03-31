import csv


class EventParser:
    def __init__(self, file_path):
        self.row_names = list()
        self.events = list()
        self.read_file(file_path)

    def read_file(self, file_path):
        with open(file_path) as f:
            data = csv.reader(f, delimiter=',')
            line_count = 0
            for row in data:
                if line_count == 0:
                    self.row_names = row
                    line_count += 1
                else:
                    self.events.append(PlayByPlayEvent(row))
                    line_count += 1

    def num_of_events(self):
        return len(self.events)

    def display_pbp(self, number):
        if 0 <= number < len(self.events):
            event = self.events[number].original_row
            displayDictionary = dict()
            displayDictionary['period'] = event[4]
            timestr = event[6].split(':')
            time = int(timestr[0])*60 + int(timestr[1])
            displayDictionary['time_left'] = time
            displayDictionary['home_desc'] = event[7]
            displayDictionary['neutral_desc'] = event[8]
            displayDictionary['away_desc'] = event[9]
            displayDictionary['score'] = event[10]
            return displayDictionary
        else:
            return None


class PlayByPlayEvent:
    def __init__(self, row):
        self.original_row = row


if __name__ == '__main__':
    e = EventParser('../match_data/0021500491.csv')
    a = 0
